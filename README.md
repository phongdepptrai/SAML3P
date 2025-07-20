# SAML3P - SAT-based Assembly Line Balancing Problem Solver

## Overview

SAML3P is a sophisticated SAT (Boolean Satisfiability) based solver for the Simple Assembly Line Balancing Problem (SALBP). This project implements advanced optimization techniques using PySAT library with Glucose3 solver to efficiently assign tasks to workstations while respecting precedence constraints and cycle time limitations.

## Features

### Core Capabilities
- **SAT-based Optimization**: Uses Boolean satisfiability solving for optimal task assignment
- **Precedence Constraints**: Handles complex task dependencies and ordering requirements
- **Cycle Time Management**: Ensures tasks complete within specified time windows
- **Workstation Balancing**: Optimizes workload distribution across multiple stations
- **Power/Weight Optimization**: Minimizes peak power consumption across all time slots

### Advanced Features
- **Staircase Constraints**: Implements sophisticated constraint reduction techniques
- **Preprocessing**: Reduces problem complexity through early/late start time analysis
- **Branch and Bound**: Iterative optimization to find minimal solutions
- **HTML Visualization**: Generates interactive Gantt charts for solution analysis
- **Multiple Instance Support**: Handles various benchmark datasets

## Project Structure

```
SAML3P/
├── SAML3P.py              # Main solver implementation
├── staircase4.py          # Enhanced version with staircase constraints
├── presedent_graph/       # Task precedence graph files (.IN2 format)
│   ├── BOWMAN.IN2
│   ├── BUXEY.IN2
│   ├── GUNTHER.IN2
│   └── ...
├── task_power/           # Task power consumption data (.txt format)
│   ├── BOWMAN.txt
│   ├── BUXEY.txt
│   ├── GUNTHER.txt
│   └── ...
├── output.txt           # Solver results and statistics
├── task_assignment.html # Generated visualization
└── README.md           # This documentation
```

## Algorithm Description

### Problem Formulation
The solver addresses the following optimization problem:
- **Minimize**: Peak power consumption across all time slots
- **Subject to**:
  - Each task assigned to exactly one workstation
  - Precedence constraints respected
  - Tasks complete within cycle time
  - No resource conflicts at any time slot

### Variables
- **X[j][k]**: Boolean variable indicating if task j is assigned to workstation k
- **S[j][t]**: Boolean variable indicating if task j starts at time t
- **A[j][t]**: Boolean variable indicating if task j is active at time t

### Constraint Types
1. **Assignment Constraints**: Each task assigned to exactly one workstation
2. **Precedence Constraints**: Task dependencies must be respected
3. **Timing Constraints**: Tasks must start and complete within valid time windows
4. **Resource Constraints**: No conflicts between tasks on same workstation
5. **Staircase Constraints**: Advanced precedence handling for improved efficiency

## Usage

### Basic Execution
```python
# Run main solver
python SAML3P.py

# Run enhanced staircase version
python staircase4.py
```

### Configuration
Modify the following parameters in the Python files:
- `n`: Number of tasks
- `m`: Number of workstations
- `c`: Cycle time
- `filename`: Select input dataset from available files

### Input Format
- **Precedence Graph Files** (.IN2): Task dependencies and processing times
- **Power Files** (.txt): Power consumption for each task

## Results and Output

### Console Output
- Constraint generation progress
- Initial solution value
- Optimization iterations
- Final solution statistics

### Generated Files
- `output.txt`: Detailed solver statistics including:
  - Number of variables and constraints
  - Solution quality (peak power)
  - Number of solutions explored
  - Execution time
- `task_assignment.html`: Interactive Gantt chart visualization

### Performance Metrics
- **Variables**: Total number of Boolean variables
- **Constraints**: Total SAT clauses generated
- **Solutions**: Number of feasible solutions found
- **Best Solutions**: Solutions improving the objective
- **Runtime**: Total execution time (with 1-hour timeout)

## Technical Details

### Preprocessing Techniques
- **Early Start Analysis**: Computes minimum start times based on precedence
- **Late Start Analysis**: Computes maximum start times working backwards
- **Infeasible Variable Elimination**: Removes impossible assignments
- **Constraint Tightening**: Reduces search space through logical deduction

### Optimization Strategy
1. Find initial feasible solution
2. Add constraints to eliminate current solution
3. Solve for improved solution
4. Repeat until no better solution exists
5. Apply timeout mechanism for large instances

### Staircase Enhancement
The `staircase4.py` version implements advanced precedence handling:
- Auxiliary variables for constraint reduction
- Improved precedence constraint encoding
- Enhanced preprocessing for better performance

## Benchmark Datasets

The project includes standard SALBP benchmark instances:
- BOWMAN, BUXEY, GUNTHER, HESKIA, JACKSON
- JAESCHKE, MANSOOR, MERTENS, MITCHELL
- ROSZIEG, SAWYER, WARNECKE

Each dataset includes:
- Task processing times
- Precedence relationships
- Power consumption values

## Dependencies

```python
pip install python-sat tabulate
```

Required Python packages:
- `python-sat`: SAT solver interface
- `tabulate`: Table formatting
- `math`, `time`, `sys`: Standard library modules

## Performance Notes

- **Scalability**: Handles instances up to 60+ tasks with 25+ workstations
- **Timeout**: 1-hour execution limit prevents excessive runtimes
- **Memory**: Efficient constraint generation minimizes memory usage
- **Preprocessing**: Significantly reduces problem complexity

## Contributing

This project implements state-of-the-art SAT-based optimization for assembly line balancing. The code demonstrates advanced techniques in:
- Constraint programming
- Boolean satisfiability solving
- Manufacturing optimization
- Algorithm engineering

## License

Research and educational use permitted. Please cite appropriately in academic work.

---

*Last updated: July 2025*