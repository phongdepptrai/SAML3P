"""
Task Assignment Validator - Final Version

This script validates the task assignment generated in task_assignment.html
to check if it satisfies all the scheduling constraints for the SAML3P problem.

Features:
- Validates all task assignments
- Checks precedence constraints (same machine vs staircase constraints)
- Calculates power consumption
- Provides detailed reporting
- Saves validation report to file
"""

import re
from bs4 import BeautifulSoup
import sys

def read_problem_data(filename, power_filename):
    """Read problem data from input file and power file"""
    n = 0
    time_list = []
    adj = []
    power_list = []
    
    # Read main problem file
    cnt = 0
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                if cnt == 0:
                    n = int(line)
                elif cnt <= n:
                    time_list.append(int(line))
                else:
                    parts = line.split(",")
                    if parts[0] != "-1" and parts[1] != "-1":
                        adj.append([int(parts[0])-1, int(parts[1])-1])  # Convert to 0-indexed
                    else:
                        break
                cnt += 1
    
    # Read power file
    with open(power_filename, 'r') as file:
        for line in file:
            power_list.append(int(line.strip()))
    
    return n, time_list, adj, power_list

def parse_html_assignment(html_file):
    """Parse the HTML file to extract task assignments"""
    with open(html_file, 'r') as file:
        content = file.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')[1:]  # Skip header row
    
    assignment = {}  # {machine: {time: task}}
    
    for i, row in enumerate(rows):
        cells = row.find_all('td')[1:]  # Skip machine label
        machine = i
        assignment[machine] = {}
        
        for t, cell in enumerate(cells):
            task_text = cell.get_text().strip()
            if task_text:
                task = int(task_text) - 1  # Convert to 0-indexed
                assignment[machine][t] = task
    
    return assignment

def validate_assignment(assignment, n, m, c, time_list, adj, power_list):
    """Validate the task assignment against all constraints"""
    errors = []
    warnings = []
    
    # 1. Check that each task is assigned to exactly one machine and runs correctly
    task_assignments = {}  # task -> (machine, start_time)
    
    # First pass: find all task occurrences
    for machine in range(m):
        for time in range(c):
            if time in assignment[machine]:
                task = assignment[machine][time]
                if task not in task_assignments:
                    # This is the first occurrence, record it
                    task_assignments[task] = (machine, time)
                else:
                    # Check if it's the same machine (tasks can't span machines)
                    recorded_machine, _ = task_assignments[task]
                    if recorded_machine != machine:
                        errors.append(f"Task {task+1} is assigned to multiple machines: {recorded_machine+1} and {machine+1}")
    
    # Check if all tasks are assigned
    for task in range(n):
        if task not in task_assignments:
            errors.append(f"Task {task+1} is not assigned to any machine")
    
    # 2. Check task duration constraints
    for task in range(n):
        if task in task_assignments:
            machine, start_time = task_assignments[task]
            duration = time_list[task]
            
            # Check if task runs for correct duration
            task_times = []
            for t in range(c):
                if t in assignment[machine] and assignment[machine][t] == task:
                    task_times.append(t)
            
            if len(task_times) != duration:
                errors.append(f"Task {task+1} runs for {len(task_times)} time units, but should run for {duration}")
            
            # Check if task runs continuously
            if task_times:
                expected_times = list(range(min(task_times), min(task_times) + duration))
                if task_times != expected_times:
                    errors.append(f"Task {task+1} does not run continuously")
            
            # Check if task fits within cycle time
            if start_time + duration > c:
                errors.append(f"Task {task+1} extends beyond cycle time")
    
    # 3. Check precedence constraints
    for i, j in adj:
        if i in task_assignments and j in task_assignments:
            machine_i, start_i = task_assignments[i]
            machine_j, start_j = task_assignments[j]
            end_i = start_i + time_list[i]
            
            # Check precedence constraints based on machine assignment
            if machine_i == machine_j:
                # Same machine: task i must finish before task j starts
                if end_i > start_j:
                    errors.append(f"Same machine precedence violation: Task {i+1} (ends at time {end_i}) must finish before Task {j+1} (starts at time {start_j}) on machine {machine_i+1}")
            else:
                # Different machines: staircase constraint
                # Task i must be assigned to a machine with lower or equal index than task j
                if machine_i > machine_j:
                    errors.append(f"Staircase constraint violation: Task {i+1} is on machine {machine_i+1} but Task {j+1} is on machine {machine_j+1}. Task {i+1} should be on a machine <= {machine_j+1}")
    
    # 4. Check no machine conflicts (no two tasks on same machine at same time)
    for machine in range(m):
        for time in range(c):
            tasks_at_time = []
            if time in assignment[machine]:
                tasks_at_time.append(assignment[machine][time])
            
            if len(set(tasks_at_time)) > 1:
                errors.append(f"Machine {machine+1} has multiple tasks at time {time+1}")
    
    # 5. Calculate maximum power consumption and timeline
    max_power = 0
    max_power_time = 0
    power_details = []
    total_power = 0
    
    for time in range(c):
        current_power = 0
        active_tasks = []
        
        for machine in range(m):
            if time in assignment[machine]:
                task = assignment[machine][time]
                current_power += power_list[task]
                active_tasks.append(task + 1)
        
        power_details.append({
            'time': time + 1,
            'power': current_power,
            'tasks': active_tasks
        })
        
        total_power += current_power
        
        if current_power > max_power:
            max_power = current_power
            max_power_time = time + 1
    
    # Calculate additional statistics
    avg_power = total_power / c if c > 0 else 0
    non_zero_power_times = [p for p in power_details if p['power'] > 0]
    
    return errors, warnings, max_power, max_power_time, power_details, task_assignments, avg_power, non_zero_power_times

def print_detailed_report(errors, warnings, max_power, max_power_time, power_details, task_assignments, avg_power, non_zero_power_times, n, m, c, adj):
    """Print a detailed validation report"""
    print("=" * 70)
    print("TASK ASSIGNMENT VALIDATION REPORT")
    print("=" * 70)
    
    print(f"\nProblem Parameters:")
    print(f"  Number of tasks (n): {n}")
    print(f"  Number of machines (m): {m}")
    print(f"  Cycle time (c): {c}")
    print(f"  Precedence constraints: {len(adj)}")
    
    print(f"\nValidation Results:")
    if not errors:
        print("  [PASS] All constraints satisfied!")
    else:
        print(f"  [FAIL] {len(errors)} constraint violations found")
    
    if warnings:
        print(f"  [WARNING] {len(warnings)} warnings")
    
    print(f"\nObjective Analysis:")
    print(f"  Maximum power consumption: {max_power}")
    print(f"  Occurs at time: {max_power_time}")
    print(f"  Average power consumption: {avg_power:.2f}")
    print(f"  Active time periods: {len(non_zero_power_times)}/{c}")
    
    if errors:
        print(f"\nCONSTRAINT VIOLATIONS:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    
    if warnings:
        print(f"\nWARNINGS:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    print(f"\nTask Assignment Summary:")
    # Group tasks by machine
    machine_tasks = {i: [] for i in range(m)}
    for task, (machine, start_time) in task_assignments.items():
        machine_tasks[machine].append((task + 1, start_time + 1))
    
    for machine in range(m):
        tasks = sorted(machine_tasks[machine], key=lambda x: x[1])  # Sort by start time
        print(f"  Machine {machine + 1}: {len(tasks)} tasks")
        for task, start_time in tasks:
            print(f"    Task {task:2d}: starts at time {start_time:3d}")
    
    # Power consumption analysis
    print(f"\nPower Consumption Analysis:")
    high_power_threshold = max_power * 0.9
    high_power_periods = [p for p in power_details if p['power'] >= high_power_threshold]
    
    print(f"  Periods with power >= {high_power_threshold:.0f}: {len(high_power_periods)} time units")
    
    # Show top power consumption periods
    sorted_power = sorted(power_details, key=lambda x: x['power'], reverse=True)
    print(f"  Top 5 highest power consumption times:")
    for i, detail in enumerate(sorted_power[:5], 1):
        if detail['power'] > 0:
            print(f"    {i}. Time {detail['time']:3d}: Power = {detail['power']:3d}, Tasks = {detail['tasks']}")

def print_summary_report(errors, max_power, n, m, c, filename):
    """Print a concise summary report"""
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    print(f"Problem: {filename}")
    print(f"Configuration: {n} tasks, {m} machines, cycle time {c}")
    
    if not errors:
        print("Status: VALID ✓")
        print("✓ All scheduling constraints are satisfied")
        print(f"✓ Objective value (max power): {max_power}")
        print("✓ The task assignment in task_assignment.html is CORRECT!")
    else:
        print("Status: INVALID ✗")
        print(f"✗ Found {len(errors)} constraint violations")
        print("✗ The task assignment needs to be corrected")

def main():
    # Configuration - modify these as needed
    filename = "HESKIA.IN2"  # Current problem file
    power_filename = "task_power/HESKIA.txt"
    html_file = "task_assignment.html"
    
    # Problem parameters (from staircase4.py)
    m = 3  # number of machines
    c = 342  # cycle time
    #
    print("Task Assignment Validator - Starting validation...")
    print(f"Input files: {filename}, {power_filename}, {html_file}")
    
    try:
        # Read problem data
        print("Reading problem data...")
        n, time_list, adj, power_list = read_problem_data(filename, power_filename)
        
        # Parse HTML assignment
        print("Parsing HTML assignment...")
        assignment = parse_html_assignment(html_file)
        
        # Validate assignment
        print("Validating assignment...")
        errors, warnings, max_power, max_power_time, power_details, task_assignments, avg_power, non_zero_power_times = validate_assignment(
            assignment, n, m, c, time_list, adj, power_list
        )
        
        # Print detailed report
        print_detailed_report(errors, warnings, max_power, max_power_time, power_details, task_assignments, avg_power, non_zero_power_times, n, m, c, adj)
        
        # Save detailed report to file
        with open("validation_report.txt", "w", encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            print_detailed_report(errors, warnings, max_power, max_power_time, power_details, task_assignments, avg_power, non_zero_power_times, n, m, c, adj)
            sys.stdout = original_stdout
        
        print(f"\nDetailed report saved to 'validation_report.txt'")
        
        # Print summary
        print_summary_report(errors, max_power, n, m, c, filename)
        
        return len(errors) == 0
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return False
    except Exception as e:
        print(f"Error during validation: {e}")
        return False

if __name__ == "__main__":
    is_valid = main()
    
    print("\n" + "=" * 50)
    if is_valid:
        print("FINAL RESULT: TASK ASSIGNMENT IS VALID! ✓")
    else:
        print("FINAL RESULT: TASK ASSIGNMENT HAS ERRORS! ✗")
    print("=" * 50)
