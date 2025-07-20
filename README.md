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
- **Pseudo-Boolean Constraints**: Direct optimization using weighted constraints (pb_staircase4)
- **Threading Support**: Robust timeout handling with multi-threaded solver execution
- **Preprocessing**: Reduces problem complexity through early/late start time analysis
- **Branch and Bound**: Iterative optimization to find minimal solutions
- **HTML Visualization**: Generates interactive Gantt charts for solution analysis
- **Multiple Instance Support**: Handles various benchmark datasets

## Project Structure

```
SAML3P/
├── SAML3P.py              # Main solver implementation
├── staircase4.py          # Enhanced version with staircase constraints
├── pb_staircase4          # Pseudo-Boolean enhanced solver with threading
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

## Solver Variants

The project includes three different solver implementations, each with increasing sophistication:

### 1. SAML3P.py (Base Implementation)
- Classic SAT-based approach using Boolean variables only
- Standard constraint generation and solving
- Basic timeout mechanism
- Suitable for smaller to medium-sized instances

### 2. staircase4.py (Enhanced Constraints)
- Implements staircase constraints for improved efficiency
- Auxiliary variables for better precedence handling
- Enhanced preprocessing techniques
- Better performance on complex precedence structures

### 3. pb_staircase4 (Pseudo-Boolean Advanced)
- **Most Advanced Version** - Recommended for large instances
- Direct pseudo-Boolean constraint encoding using PySAT's PBEnc
- Multi-threaded solver execution with robust timeout handling
- Weighted constraint optimization: ∑(wᵢ × Aᵢ,ₜ) ≤ bound
- Enhanced memory management and variable tracking
- Superior scalability for complex industrial instances

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

The solver uses different variable encoding schemes depending on the implementation:

#### Base Variables (All Implementations)
- **X[j][k]**: Boolean variable indicating if task j is assigned to workstation k
- **A[j][t]**: Boolean variable indicating if task j is active at time t

#### SAML3P.py Variables
- **S[j][t]**: Boolean variable indicating if task j starts at time t
- Variable indexing: X uses `j*m+k+1`, A uses `m*n + j*c + t + 1`, S uses `m*n + c*n + j*c + t + 1`

#### staircase4.py Variables
- **S[j][t]**: Boolean variable indicating if task j starts at time t (variable length per task)
- **R[j][k]**: Auxiliary staircase variables for precedence constraints
- Variable indexing: Optimized S arrays with `c - time_list[j] + 1` variables per task
- Auxiliary variables managed through `var_map` dictionary

#### pb_staircase4 Variables
- **S[j][t]**: Boolean variable indicating if task j starts at time t (optimized indexing)
- **T[j][t]**: Auxiliary temporal variables for staircase constraints
- Enhanced variable management with dynamic counter tracking
- Pseudo-Boolean encoding creates additional auxiliary variables automatically

### Constraint Types

The implementations use different constraint formulations:

#### Common Constraints (All Versions)
1. **Assignment Constraints**: Each task assigned to exactly one workstation
2. **Precedence Constraints**: Task dependencies must be respected
3. **Timing Constraints**: Tasks must start and complete within valid time windows
4. **Resource Constraints**: No conflicts between tasks on same workstation

#### SAML3P.py Specific Constraints
- Standard Boolean clauses for all constraints
- Direct encoding of timing relationships
- Basic precedence handling with O(n²m²t²) complexity

#### staircase4.py Enhanced Constraints
5. **Staircase Constraints**: Advanced precedence handling using auxiliary R variables
   - R[j][k] → R[j][k+1] (monotonicity)
   - X[j][k] ↔ (R[j][k] ∧ ¬R[j][k-1]) (assignment encoding)
   - Reduced complexity for precedence constraints

#### pb_staircase4 Pseudo-Boolean Constraints
6. **Weighted Objective Constraints**: Direct pseudo-Boolean encoding
   - ∑(wᵢ × A[i][t]) ≤ bound for all time slots t
   - Automatic auxiliary variable generation by PBEnc
   - More efficient than clause-based objective handling

## Usage

### Basic Execution
```python
# Run main solver
python SAML3P.py

# Run enhanced staircase version
python staircase4.py

# Run pseudo-Boolean enhanced version with threading
python pb_staircase4
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
- **Variables**: Total number of Boolean variables (varies by implementation)
  - SAML3P.py: n×m + 2×n×c variables
  - staircase4.py: n×m + n×c + Σ(c - time_list[j] + 1) + auxiliary R variables
  - pb_staircase4: Optimized variable count + PBEnc auxiliary variables
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

### Pseudo-Boolean Enhancement
The `pb_staircase4` version provides the most advanced implementation:
- **Direct Optimization**: Uses PySAT's PBEnc for weighted constraint encoding
- **Threading Support**: Robust timeout handling with multi-threaded execution
- **Improved Performance**: More efficient handling of objective function constraints
- **Better Scalability**: Handles larger instances with enhanced memory management
- **Weighted Constraints**: Direct encoding of ∑(wᵢ × Aᵢ,ₜ) ≤ bound constraints

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
pip install python-sat tabulate numpy
```

Required Python packages:
- `python-sat`: SAT solver interface with pseudo-Boolean constraint support
- `tabulate`: Table formatting
- `numpy`: Numerical computing (for pb_staircase4)
- `math`, `time`, `sys`, `threading`: Standard library modules

## Performance Notes

- **Scalability**: Handles instances up to 60+ tasks with 25+ workstations
- **Timeout**: 1-hour execution limit with robust threading (pb_staircase4)
- **Memory**: Efficient constraint generation minimizes memory usage
- **Preprocessing**: Significantly reduces problem complexity
- **Solver Performance**: 
  - `SAML3P.py`: Best for instances < 30 tasks
  - `staircase4.py`: Optimal for 30-50 tasks with complex precedence
  - `pb_staircase4`: Recommended for 50+ tasks and industrial applications

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