import re # regex patterns for parsing move syntax
import ast # safe evaluation of list strings
import json # standard json library
import time # time-related functions
from collections import Counter # counting votes
from openai import OpenAI # client for local LLM connection

# ==========================================
# CONFIGURATION
# ==========================================
# Connect to LM Studio
# Sets up connection to local server on port 1234
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Paper Parameters
K_THRESHOLD = 10   # The 'k' in "First-to-ahead-by-k". Higher = more accurate but slower.
NUM_DISKS = 3     # Start small (3-4) for local testing. The paper uses 20.
MAX_TOKENS = 750  # Red-flag cutoff [cite: 521]

# ==========================================
# 1. THE ENVIRONMENT (Logic & State)
# ==========================================
class HanoiEnv:
    def __init__(self, num_disks):
        # Initialize pegs with disks on peg 0
        self.num_disks = num_disks
        # State is a list of 3 lists. Peg 0 has all disks (1 is smallest).
        self.state = [list(range(num_disks, 0, -1)), [], []]
        self.history = [] # To track "previous move" for the prompt
        
    def get_legal_moves(self):
        # Helper to validate logic internally
        moves = []
        top_disks = [peg[-1] if peg else float('inf') for peg in self.state]
        for start_peg, disk in enumerate(top_disks):
            if disk == float('inf'): continue
            for end_peg, target_top in enumerate(top_disks):
                if start_peg != end_peg and disk < target_top:
                    moves.append([disk, start_peg, end_peg])
        return moves

    def apply_move(self, move):
        # Executes a move: [disk, from_peg, to_peg]
        disk, start, end = move
        # Validation: Check if move is legal
        if not self.state[start] or self.state[start][-1] != disk:
            return False, "Invalid: Disk not on top of start peg."
        if self.state[end] and self.state[end][-1] < disk:
            return False, "Invalid: Cannot place larger disk on smaller."
        
        # Execute: Pop from start, push to end
        self.state[start].pop()
        self.state[end].append(disk)
        self.history.append(move)
        return True, "Success"

    def is_solved(self):
        # Check if all disks are on the last peg
        return len(self.state[2]) == self.num_disks

    def __str__(self):
        return str(self.state)

# ==========================================
# 2. PROMPTS & RED-FLAGGING (Parser)
# ==========================================
# Prompts adapted directly from Appendix C [cite: 714]

SYSTEM_PROMPT = """You are a helpful assistant. Solve this puzzle for me.
There are three pegs and n disks of different sizes. The disks are numbered 1 (smallest) to n (largest).
Rules:
1. Only one disk can be moved at a time.
2. A larger disk may not be placed on top of a smaller disk.
3. Your goal is to determine the SINGLE NEXT MOVE based on the current state.

Format your response EXACTLY as follows:
'''
move = [disk_id, from_peg, to_peg]
next_state = [[...], [...], [...]]
'''
"""

def construct_user_prompt(current_state, previous_move):
    # Dynamically builds prompt with state and strategy
    # Strategy injection as per Section 4.1 
    strategy_hint = """
    Strategy:
    If the previous move did NOT move disk 1, move disk 1 clockwise (0->1->2->0).
    If the previous move DID move disk 1, make the only legal move that does NOT involve disk 1.
    """
    
    prev_str = str(previous_move) if previous_move else "None (First Move)"
    
    return f"""
    {strategy_hint}
    
    Previous move: {prev_str}
    Current State: {current_state}
    
    Based on the previous move and current state, find the next move and resulting state.
    """

def red_flag_parser(response_text):
    """
    Implements Red-Flagging[cite: 367].
    Discards output if:
    1. It cannot be parsed (Bad Format).
    2. Logic is hallucinatory (structure doesn't match).
    """
    try:
        # Extract content between ''' if present, or raw text
        clean_text = response_text.replace("'''", "")
        
        # Regex to find move and state patterns
        move_match = re.search(r"move\s*=\s*(\[[0-9, ]+\])", clean_text)
        state_match = re.search(r"next_state\s*=\s*(\[\[.*\]\])", clean_text, re.DOTALL)
        
        if not move_match or not state_match:
            return None, "Red Flag: Missing move or state format"
            
        # Safe evaluation of list strings
        move = ast.literal_eval(move_match.group(1))
        next_state = ast.literal_eval(state_match.group(1))
        
        # Basic Type Checks
        if not isinstance(move, list) or len(move) != 3:
            return None, "Red Flag: Move format invalid"
            
        return (tuple(move), str(next_state)), None # Return as hashable tuple for voting
        
    except Exception as e:
        return None, f"Red Flag: Parsing Exception - {str(e)}"

# ==========================================
# 3. MAKER AGENT (Voting Logic)
# ==========================================

def get_agent_vote(current_state, previous_move):
    """Call LM Studio for a single vote"""
    try:
        # Request completion from local model
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b", # LM Studio usually ignores this, but it's required
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": construct_user_prompt(current_state, previous_move)}
            ],
            temperature=0.7, # Higher temp is needed for independent samples [cite: 587]
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return ""

def calculate_error_rate(errors, total):
    if total == 0: return 0.0
    return errors / total

def solve_step_maker(env, stats, k=K_THRESHOLD):
    """
    Implements First-to-ahead-by-k Voting[cite: 179].
    Updates stats dictionary with error counts.
    """
    votes = Counter()
    attempts = 0
    max_attempts = 50 # Increased safety break for higher K
    
    # Get legal moves for logic validation
    legal_moves = {tuple(m) for m in env.get_legal_moves()}
    
    print(f"\n--- Step {len(env.history) + 1} ---")
    
    while attempts < max_attempts:
        # 1. Sample Agent
        stats['total_samples'] += 1
        raw_response = get_agent_vote(env.state, env.history[-1] if env.history else None)
        
        # 2. Red-Flagging (Discard if invalid)
        parsed_result, error = red_flag_parser(raw_response)
        
        # Check for Red Flags (Syntax)
        if parsed_result is None:
            stats['error_count'] += 1
            print(f"Sample discarded: {error}")
            attempts += 1
            continue 
            
        move_tuple, state_str = parsed_result
        
        # Check for Hallucinated Moves (Logic)
        if move_tuple not in legal_moves:
            stats['error_count'] += 1
            print(f"Sample discarded: Illegal Move {move_tuple} not in {legal_moves}")
            attempts += 1
            continue
        
        # 3. Add to Vote
        votes[move_tuple] += 1
        attempts += 1
        
        # 4. Check 'First-to-ahead-by-k' Condition
        if len(votes) >= 1:
            most_common = votes.most_common(2)
            leader_move, leader_count = most_common[0]
            
            if len(most_common) == 1:
                runner_up_count = 0
            else:
                _, runner_up_count = most_common[1]
            
            margin = leader_count - runner_up_count
            
            # Calculate and show current step error rate (optional, but useful)
            # We use the global stats here which is cumulative, 
            # OR we could track local step stats if preferred. 
            # Let's show cumulative to match the request context.
            cumulative_err_rate = calculate_error_rate(stats['error_count'], stats['total_samples'])
            print(f"Votes: {dict(votes)} | Margin: {margin}/{k} | Cumulative Error Rate: {cumulative_err_rate:.1%}")
            
            # Stop if margin requirement met
            if margin >= k:
                return list(leader_move)

    print(f"Warning: Max attempts reached. Cumulative Error Rate: {calculate_error_rate(stats['error_count'], stats['total_samples']):.1%}")
    
    # Fallback if no valid votes were ever collected
    if not votes:
        print("CRITICAL: No valid moves found after max attempts. Returning None.")
        return None
        
    return list(votes.most_common(1)[0][0])

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
def main():
    game = HanoiEnv(NUM_DISKS)
    print(f"Starting MAKER with {NUM_DISKS} disks on Local LM Studio.")
    print(f"Initial State: {game.state}")

    # Global Statistics
    stats = {'total_samples': 0, 'error_count': 0}

    step_count = 0
    # Loop until solved
    while not game.is_solved():
        # Perform MAKER step: get move from AI agent
        chosen_move = solve_step_maker(game, stats, K_THRESHOLD)
        
        if chosen_move is None:
            print("Agent failed to produce any valid move. Terminating.")
            break

        # Execute in Environment
        success, msg = game.apply_move(chosen_move)
        
        step_count += 1
        if success:
            print(f"Executed Move: {chosen_move}")
            print(f"New State: {game.state}")
        else:
            print(f"CRITICAL ERROR: Agent selected illegal move {chosen_move}: {msg}")
            break
            
    if game.is_solved():
        final_error_rate = calculate_error_rate(stats['error_count'], stats['total_samples'])
        system_accuracy = 1.0 - final_error_rate
        
        print(f"\nSOLVED in {step_count} steps!")
        print(f"Final Statistics:")
        print(f"  Total Samples: {stats['total_samples']}")
        print(f"  Total Errors:  {stats['error_count']}")
        print(f"  Final Error Rate: {final_error_rate:.1%}")
        print(f"  System Accuracy:  {system_accuracy:.1%}")

if __name__ == "__main__":
    main()