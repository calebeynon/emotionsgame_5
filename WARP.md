# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an oTree behavioral economics experiment implementing a public goods game with chat functionality. The experiment studies how communication affects contribution behavior across multiple rounds with strategic regrouping between supergames.

## Key Commands

### Development Environment
- **Run development server**: `rye run otree devserver`
- **Run tests**: `rye run otree test` (runs all app tests)
- **Run specific app test**: `rye run otree test <app_name>` (e.g., `rye run otree test supergame1`)
- **Reset database**: `rye run otree resetdb`
- **Create session**: `rye run otree create_session public_goods 16`

### Production/Deployment
- **Run production server**: `rye run otree prodserver`
- **Zip for deployment**: `rye run otree zip`

## Experiment Architecture

### Game Flow Structure
The experiment follows this sequence:
1. **introduction**: Instructions and comprehension quiz
2. **supergame1**: 3 rounds of public goods game with initial grouping
3. **supergame2**: 4 rounds with first regrouping  
4. **supergame3**: 4 rounds with second regrouping
5. **supergame4**: 4 rounds with final regrouping
6. **finalresults**: Survey and payment information

### Grouping Strategy
- 16 participants total, 4 players per group
- Each supergame uses different grouping matrices defined in `creating_session()` functions
- Strategic regrouping between supergames to study repeated game effects
- Participant labels (A-R) used for anonymization, loaded from `participant_labels.txt`

### Game Mechanics
- **Endowment**: 25 points per round
- **Multiplier**: 0.4 (40% of total contributions returned to group)
- **Payoff formula**: `individual_payoff = endowment - contribution + (group_total * multiplier)`
- **Chat functionality**: Timed chat periods with previous round feedback

## Code Architecture

### App Structure
Each supergame app (`supergame1`-`supergame4`) follows identical patterns:
- **Models**: `Player.contribution`, `Group.total_contribution/individual_share`
- **Key Functions**: 
  - `creating_session()`: Defines group matrices
  - `set_payoffs()`: Calculates payoffs and stores in `participant.vars['payoff_list']`
  - `set_payoffs_0()`: Resets payoff tracking between supergames

### Page Sequence Pattern
Standard flow per supergame:
1. `StartPage`: Shows group member labels (round 1 only)
2. `ChatFirst`: Extended chat for first round (120s timeout)
3. `Chat`: Subsequent round chats with previous results (30s timeout)
4. `Contribute`: Main decision page
5. `Results`: Round outcomes with cumulative table
6. `RegroupingMessage`: End-of-supergame transition

### Data Management
- **Payoff tracking**: Each supergame stores cumulative payoffs in `participant.vars['payoff_sum_X']`
- **Template variables**: Complex logic in `vars_for_template()` methods for displaying previous round information
- **Final payment**: Calculated from total participant payoffs plus $7.50 participation fee

## Development Notes

### Chat Implementation
- Uses oTree's built-in `{{ chat }}` functionality with participant labels as nicknames
- Previous round feedback displayed during chat phases (except first round)
- Timeout mechanisms prevent infinite waiting

### Testing Considerations
- Session config set for 16 demo participants matching production requirements
- Room-based sessions using participant label files
- Each app includes `tests.py` for unit testing

### Template Customization
- Global templates in `_templates/global/` extend oTree base
- Custom styling in individual app templates for readability
- Table-based results display with round-by-round contribution history

## Session Configuration

The main session config "public_goods" is configured in `settings.py`:
- Requires exactly 16 participants
- Uses room-based assignment with participant labels
- Real-world currency conversion: $0.10 per point + $7.50 participation fee
