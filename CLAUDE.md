# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains multiple Python applications:

1. **Chess Game** (`chess.py`) - A 6x6 chess variant with economic gameplay elements built with pygame
2. **Streamlit Apps** (`X.py`, `number_magic.py`) - Interactive web applications

## Running the Applications

### Chess Game
```bash
python chess.py
```
- Requires pygame (auto-installed via requirements.txt)
- Features: 6x6 board, piece purchasing system, AI opponent, dog companion images, background music

### Streamlit Applications
```bash
streamlit run [filename.py]
```
- `X.py` - General streamlit application
- `number_magic.py` - Number-based interactive application

## Chess Game Architecture

### Core Classes

**ChessGame** - Main game logic class
- Board management (6x6 grid)
- Turn-based gameplay (white vs black)
- Economic system (money, income per turn)
- Piece purchasing with randomized rewards
- Employee hiring and class change mechanics
- UI state management

**SimGame** - Lightweight simulation class for AI computations
- Identical game logic to ChessGame but optimized for speed
- Used exclusively by AI for move evaluation
- No UI elements (dog images, sounds, display)

### Game Mechanics

**Economic System:**
- Starting money: 50G each player
- Income per turn: 10G base (+2G per employee hired)
- Piece purchasing options:
  - Normal (20G): 90% pawn, 3.3% each for rook/knight/bishop, 0.1% queen
  - Rare (30G): 50% pawn, 15% each for rook/knight/bishop, 5% queen  
  - Epic (40G): 10% pawn, 25% each for rook/knight/bishop, 15% queen
- Territory bonuses: +5G for capturing pieces in enemy territory
- Employee hiring: 40G for +2G income per turn
- Class changes: 70G to upgrade pieces (pawn→knight→bishop→rook)

**AI Implementation:**
- Alpha-beta pruning with transposition table
- Search depth: 4 levels
- Evaluation factors: piece values, position bonuses, material balance, money advantage
- Move ordering for pruning efficiency
- Located in `cpu_play_turn()` method (lines 693-836)

### Key Files and Directories

- `chess_pieces/` - PNG images for all chess pieces (12 files: w/b + p/r/n/b/q/k)
- `dog/` - Companion dog images for different game states
- `sound/` - Background music and sound effects
- `haikei.png` - Background image
- `requirements.txt` - Python dependencies (streamlit, pygame auto-installed)

### Development Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Test the chess game:**
```bash
python chess.py
```

**Run streamlit apps:**
```bash
streamlit run X.py
streamlit run number_magic.py
```

## Git Workflow

This project uses standard git workflow on the `main` branch:

```bash
git add .
git commit -m "descriptive message"
git push origin main
```

If encountering push conflicts:
```bash
git pull origin main
```

## AI Analysis

The current AI implementation has several limitations:

1. **Limited search depth** - Only 4 levels deep, restricting tactical vision
2. **Simplistic evaluation** - Basic piece values + position bonuses
3. **No opening/endgame knowledge** - Uniform strategy throughout game phases
4. **Economic strategy gaps** - Suboptimal piece purchasing and investment decisions

The AI uses alpha-beta pruning which is suitable for this game's deterministic move evaluation, but the economic and random elements (piece purchasing) suggest **Monte Carlo Tree Search (MCTS)** could be more effective for handling uncertainty and complex strategic decisions.

## Code Conventions

- Japanese comments and variable names are used throughout
- pygame for graphics and input handling
- Object-oriented design with clear separation between game logic and AI
- Consistent indentation and naming patterns
- Error handling for missing assets (images, sounds)