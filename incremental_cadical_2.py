from math import inf
import math
import re
import time
import signal
import json
import subprocess
import os
import pandas as pd
from datetime import datetime
import sys
import csv
from pysat.solvers import Cadical195
from pysat.pb import PBEnc

# Global variables
n = 25
m = 6  
c = 25
val = 0
cons = 0
sol = 0
solbb = 0
type = 1
best_result = None
current_instance_id = 0
start_time_global = 0

# Initialize global data structures
filename = ""
W = []
neighbors = []
reversed_neighbors = []
visited = []
toposort = []
clauses = []
time_list = []
adj = []
forward = []
var_map = {}
var_counter = 0

# Instance definitions - consolidated from file_name and file_name1
INSTANCES = [
    # Easy families (0-38)
    ["MERTENS", 6, 6], ["MERTENS", 2, 18], ["MERTENS", 5, 7], ["MERTENS", 5, 8], ["MERTENS", 3, 10], ["MERTENS", 2, 15],
    ["BOWMAN", 5, 20],
    ["JAESCHKE", 8, 6], ["JAESCHKE", 3, 18], ["JAESCHKE", 6, 8], ["JAESCHKE", 4, 10], ["JAESCHKE", 3, 18],
    ["JACKSON", 8, 7], ["JACKSON", 3, 21], ["JACKSON", 6, 9], ["JACKSON", 5, 10], ["JACKSON", 4, 13], ["JACKSON", 4, 14],
    ["MANSOOR", 4, 48], ["MANSOOR", 2, 94], ["MANSOOR", 3, 62],
    ["MITCHELL", 8, 14], ["MITCHELL", 3, 39], ["MITCHELL", 8, 15], ["MITCHELL", 5, 21], ["MITCHELL", 5, 26], ["MITCHELL", 3, 35],
    ["ROSZIEG", 10, 14], ["ROSZIEG", 4, 32], ["ROSZIEG", 6, 25], ["ROSZIEG", 8, 16], ["ROSZIEG", 8, 18], ["ROSZIEG", 6, 21],
    ["HESKIA", 8, 138], ["HESKIA", 3, 342], ["HESKIA", 5, 205], ["HESKIA", 5, 216], ["HESKIA", 4, 256], ["HESKIA", 4, 324],
    
    # Hard families (39+)
    ["BUXEY", 7, 47], ["BUXEY", 8, 41], ["BUXEY", 11, 33], ["BUXEY", 13, 27], ["BUXEY", 12, 30], ["BUXEY", 7, 54], ["BUXEY", 10, 36],
    ["SAWYER", 14, 25], ["SAWYER", 7, 47], ["SAWYER", 8, 41], ["SAWYER", 12, 30], ["SAWYER", 13, 27], ["SAWYER", 11, 33], ["SAWYER", 10, 36], ["SAWYER", 7, 54], ["SAWYER", 5, 75],
    ["GUNTHER", 9, 54], ["GUNTHER", 9, 61], ["GUNTHER", 14, 41], ["GUNTHER", 12, 44], ["GUNTHER", 11, 49], ["GUNTHER", 8, 69], ["GUNTHER", 7, 81],
    ["WARNECKE", 25, 65], ["WARNECKE", 31, 54], ["WARNECKE", 29, 56], ["WARNECKE", 29, 58], ["WARNECKE", 27, 60], ["WARNECKE", 27, 62], ["WARNECKE", 24, 68], ["WARNECKE", 23, 71], ["WARNECKE", 22, 74], ["WARNECKE", 21, 78], ["WARNECKE", 20, 82], ["WARNECKE", 19, 86], ["WARNECKE", 17, 92], ["WARNECKE", 17, 97], ["WARNECKE", 15, 104], ["WARNECKE", 14, 111],
    ["LUTZ2", 49, 11], ["LUTZ2", 44, 12], ["LUTZ2", 40, 13], ["LUTZ2", 37, 14], ["LUTZ2", 34, 15], ["LUTZ2", 31, 16], ["LUTZ2", 29, 17], ["LUTZ2", 28, 18], ["LUTZ2", 26, 19], ["LUTZ2", 25, 20], ["LUTZ2", 24, 21]
]

# Filter instances if timeout file exists
if os.path.exists("incremental_cadical_timeout.txt"):
    with open("incremental_cadical_timeout.txt", "r") as f:
        timeout_instances = [line.strip() for line in f if line.strip()]
    INSTANCES = [instance for instance in INSTANCES if instance[0] in timeout_instances]

# Signal handler for graceful interruption
def handle_interrupt(signum, frame):
    print(f"\nReceived interrupt signal {signum}. Saving current best solution.")
    
    if best_result:
        result = best_result.copy()
        result['Status'] = 'TIMEOUT'
        result['Runtime'] = time.time() - start_time_global
    else:
        result = {
            'Instance': INSTANCES[current_instance_id][0] if current_instance_id < len(INSTANCES) else 'Unknown',
            'n': n, 'm': m, 'c': c,
            'Variables': 0, 'SoftClauses': 0, 'HardClauses': 0, 'OptimalValue': 0,
            'Status': 'TIMEOUT', 'Runtime': time.time() - start_time_global
        }
    
    # Save result as JSON for the controller to pick up
    with open(f'results_incremental_cadical_{current_instance_id}.json', 'w') as f:
        json.dump(result, f)
    
    print(f"Results saved for instance {current_instance_id}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_interrupt)
signal.signal(signal.SIGINT, handle_interrupt)

def reset_globals(idx):
    """Reset global variables for new instance"""
    global n, m, c, val, cons, sol, solbb, type, filename, W, neighbors, reversed_neighbors
    global visited, toposort, clauses, time_list, adj, forward, var_map, var_counter, current_instance_id
    
    current_instance_id = idx
    m = INSTANCES[idx][1]
    c = INSTANCES[idx][2]
    val = cons = sol = solbb = 0
    type = 1
    var_counter = 0
    var_map = {}
    filename = INSTANCES[idx][0] + ".IN2"
    
    # Load power values
    W = [int(line.strip()) for line in open(f'task_power/{INSTANCES[idx][0]}.txt')]
    
    # Reset data structures
    neighbors = [[0 for _ in range(200)] for _ in range(200)]
    reversed_neighbors = [[0 for _ in range(200)] for _ in range(200)]
    visited = [False for _ in range(200)]
    toposort = []
    clauses = []
    time_list = []
    adj = []
    forward = [0 for _ in range(200)]

def read_input():
    """Read precedence graph input file"""
    global n, adj, neighbors, reversed_neighbors, filename, time_list, forward
    cnt = 0
    with open(f'presedent_graph/{filename}', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                if cnt == 0:
                    n = int(line)
                elif cnt <= n:
                    time_list.append(int(line))
                else:
                    parts = line.split(",")
                    if parts[0] != "-1" and parts[1] != "-1":
                        i, j = int(parts[0])-1, int(parts[1])-1
                        adj.append([i, j])
                        neighbors[i][j] = 1
                        reversed_neighbors[j][i] = 1
                    else:
                        break
                cnt += 1

def generate_variables(n, m, c):
    """Generate decision variables"""
    x = [[j*m+i+1 for i in range(m)] for j in range(n)]
    a = [[m*n + j*c + i + 1 for i in range(c)] for j in range(n)]
    s = []
    cnt = m*n + c*n + 1
    for j in range(n):
        tmp = []
        for i in range(c - time_list[j] + 1):
            tmp.append(cnt)
            cnt += 1
        s.append(tmp)
    return x, a, s

def dfs(v):
    """DFS for topological sorting"""
    visited[v] = True
    for i in range(n):
        if neighbors[v][i] == 1 and not visited[i]:
            dfs(i)
    toposort.append(v)

def get_var(name, *args):
    """Get or create variable"""
    global var_counter
    key = (name,) + args
    if key not in var_map:
        var_counter += 1
        var_map[key] = var_counter
    return var_map[key]

def set_var(var, name, *args):
    """Set variable mapping"""
    key = (name,) + args
    if key not in var_map:
        var_map[key] = var
    return var_map[key]

def preprocess(n,m,c,time_list,adj):
    earliest_start = [[-9999999 for _ in range(m)] for _ in range(n)]
    latest_start = [[99999999 for _ in range(m)] for _ in range(n)]
    ip1 = [[0 for _ in range(m)] for _ in range(n)]
    test_ip1 = [[0 for _ in range(m)] for _ in range(n)]
    ip2 = [[[0 for _ in range(c)] for _ in range(m)] for _ in range(n)]
    # Compute earliest possible starting date and assigned workstation
    for i in range(n):
        if not visited[i]:
            dfs(i)
    toposort.reverse()
    # print(toposort)
    for j in toposort:
        k = 0
        earliest_start[j][k] = 0
        for i in range(n):
            if neighbors[i][j] == 1:

                earliest_start[j][k] = max(earliest_start[j][k], earliest_start[i][k] + time_list[i])

                while(earliest_start[j][k] > c - time_list[j]):
                    ip1[j][k] = 1
                    # print('X '+str(j+1)+' '+str(k+1))
                    k = k + 1
                    earliest_start[j][k] = max(0, earliest_start[i][k] + time_list[i])

                if earliest_start[j][k] <= c - time_list[j] :
                    for t in range(earliest_start[j][k]):
                        
                        if(ip2[j][k][t] == 0):
                            # with open("output.txt", "a") as output_file: 
                            #     sys.stdout = output_file  
                            #     print(j+1, k+1, t, file=output_file) 
                            ip2[j][k][t] = 1
    toposort.reverse()
    for j in toposort:
        k = m-1
        latest_start[j][k] = c - time_list[j]
        for i in range(n):
            if(neighbors[j][i] == 1): 
                latest_start[j][k] = min(latest_start[j][k], latest_start[i][k] - time_list[j])
                while(latest_start[j][k] < 0):
                    ip1[j][k] = 1
                    # print('X '+str(j+1)+' '+str(k+1))
                    k = k - 1
                    latest_start[j][k] = min(c - time_list[j], latest_start[i][k] - time_list[j])
                
                if(latest_start[j][k] >= 0):
                        for t in range(latest_start[j][k] + 1, c):
                            
                            if(ip2[j][k][t] == 0):
                                # with open("output.txt", "a") as output_file: 
                                #     sys.stdout = output_file
                                #     print(j+1, k+1, t, file=output_file) 
                                ip2[j][k][t] = 1
    
    # sys.stdout = sys.__stdout__


    # for j in range(n):
    #     for k in range(m):
    #         for t in range(c):
                # if(ip1[j][k] == 1):
                #     continue
                # if(j == 11 or j == 14):
                #     print(f"task {j+1} in machine {k+1} time {t+1}: {ip2[j][k][t]}")
                # if(j == 0 and k == 2):
                #     print(f"task {j+1} in machine {k+1} time {t+1}: {ip2[j][k][t]}")
    # print(ip2)
    return ip1,ip2

def generate_clauses(n,m,c,time_list,adj,ip1,ip2,X,S,A):
    # #test
    # clauses.append([X[11 - 1][2 - 1]])
    global clauses
    global var_map
    global var_counter
    #staircase constraints
    for j in range(n):
        
        set_var(X[j][0], "R", j, 0)
        for k in range(1,m-1):
            if ip1[j][k] == 1:
                set_var(get_var("R", j, k-1), "R", j, k)
            else:
                clauses.append([-get_var("R", j, k-1), get_var("R", j, k)])
                clauses.append([-X[j][k], get_var("R", j, k)])
                clauses.append([-X[j][k], -get_var("R", j, k-1)])
                clauses.append([X[j][k], get_var("R", j, k-1), -get_var("R", j, k)])
        # last machine
        if ip1[j][m-1] == 1:
            clauses.append([get_var("R", j, m-2)])
        else:
            clauses.append([get_var("R", j, m-2), X[j][m-1]])
            clauses.append([-get_var("R", j, m-2), -X[j][m-1]])
        

    for (i,j) in adj:
        for k in range(m-1):
            if ip1[i][k+1] == 1:
                continue
            clauses.append([-get_var("R", j, k), -X[i][k+1]])

    # T[j][t] represents "task j starts at time t or earlier"
    for j in range(n):
        last_t = c-time_list[j]
        
        # Special case: Full cycle tasks (only one feasible start time: t=0)
        if last_t == 0:
            # Force the task to start at t=0 (equivalent to original constraint #4)
            clauses.append([S[j][0]])
        else:
            # First time slot
            set_var(S[j][0], "T", j, 0)
            
            # Intermediate time slots
            for t in range(1, last_t):
                clauses.append([-get_var("T", j, t-1), get_var("T", j, t)]) # T[j][t-1] -> T[j][t]
                clauses.append([-S[j][t], get_var("T", j, t)]) # S[j][t] -> T[j][t]
                clauses.append([-S[j][t], -get_var("T", j, t-1)]) # S[j][t] -> ¬T[j][t-1]
                clauses.append([S[j][t], get_var("T", j, t-1), -get_var("T", j, t)]) # T[j][t] -> (T[j][t-1] ∨ S[j][t])
            
            # Last time slot (ensures at least one start time)
            clauses.append([get_var("T", j, last_t-1), S[j][last_t]])
            clauses.append([-get_var("T", j, last_t-1), -S[j][last_t]])
    #7
    for i in range(n-1):
        for j in range(i+1,n):
            for k in range (m):
                if ip1[i][k] == 1 or ip1[j][k] == 1 :
                    continue
                for t in range(c):
                    clauses.append([-X[i][k], -X[j][k], -A[i][t], -A[j][t]])
    for j in range(n):
        for t in range (c-time_list[j]+1):
            for l in range (time_list[j]):
                if(time_list[j] >= c/2 and t+l >= c-time_list[j] and t+l < time_list[j]):
                    continue
                clauses.append([-S[j][t],A[j][t+l]])
    for i,j in adj:
        for k in range(m):
            if ip1[i][k] == 1 or ip1[j][k] == 1:
                continue
            left_bound = time_list[i] - 1
            right_bound = c - time_list[j]
            clauses.append([-X[i][k], -X[j][k], -get_var("T", j, left_bound)])
            for t in range (left_bound + 1, right_bound):
                t_i = t - time_list[i]+1
                clauses.append([-X[i][k], -X[j][k], -get_var("T", j, t), -S[i][t_i]])
            for t in range (max(0,right_bound - time_list[i] + 1), c - time_list[i] + 1):
                clauses.append([-X[i][k], -X[j][k], -S[i][t], -get_var("T",j,c-time_list[j]-1)])

    # #10
    for j in range(n):
        for k in range(m):
            if ip1[j][k] == 1:
                clauses.append([-X[j][k]])
                continue
                # print("constraint ", j+1, k+1)
            #11
            for t in range(c - time_list[j] +1):
                if ip2[j][k][t] == 1:
                    clauses.append([-X[j][k], -S[j][t]])
                    # print("constraint ", j+1, k+1, t)
    
    #12 
    for j in range(n):
        if(time_list[j] >= c/2):
            for t in range(c-time_list[j],time_list[j]):
                clauses.append([A[j][t]])
    print("12 constraints:", len(clauses))
    return clauses

def solve(solver):
    """Solve SAT instance"""
    return solver.get_model() if solver.solve() else None

def get_value(solution, best_value):
    """Calculate objective value from solution"""
    if solution is None:
        return 100, 0, []
    
    x = [[solution[j*m+i] for i in range(m)] for j in range(n)]
    a = [[solution[m*n + j*c + i] for i in range(c)] for j in range(n)]
    
    cnt = m*n + c*n
    s = []
    for j in range(n):
        tmp = []
        for i in range(c - time_list[j] + 1):
            tmp.append(solution[cnt])
            cnt += 1
        s.append(tmp)
    
    value = lowval = 0
    constraints = []
    
    for t in range(c):
        tmp = 0
        station = []
        for j in range(n):
            if a[j][t] > 0:
                for l in range(max(0, t-time_list[j]), t+1):
                    if l < len(s[j]) and s[j][l] > 0:
                        tmp += W[j]
                        station.append(j+1)
                        break
        
        if tmp > value:
            value = tmp
        if tmp < lowval or lowval == 0:
            lowval = tmp
        
        if tmp >= min(best_value, value):
            constraints.append(station)
    
    unique_constraints = list(map(list, set(map(tuple, constraints))))
    return value, lowval, unique_constraints

def get_value2(n, m, c, model, W, UB=0):
    """Calculate objective value for optimization"""
    ans_map = [[0 for _ in range(c)] for _ in range(m + 1)]
    start_B = n * m
    
    for i in range(m):
        for j in range(c):
            for k in range(n):
                if model[k*m + i] > 0 and model[start_B + k*c + j] > 0:
                    ans_map[i][j] = W[k]
    
    for i in range(c):
        ans_map[m][i] = sum(ans_map[j][i] for j in range(m))
    
    peak = max(ans_map[m][i] for i in range(c))
    return ans_map, peak

def generate_inagural(n, m, c, X, S, A, W, UB, LB, clauses, var_counter, solver):
    """Generate constraints for incremental optimization"""
    soft_clauses = []
    U = []
    
    for i in range(LB + 1, UB):
        U.append(var_counter + 1)
        var_counter += 1
        soft_clauses.append([[-var_counter], 1])
    
    for i in range(1, len(U)):
        clause = [-U[i], U[i-1]]
        clauses.append(clause)
        solver.add_clause(clause)
    
    var = var_counter + 1
    for t in range(c):
        variables = [-U[i] for i in range(len(U))] + [A[i][t] for i in range(n)]
        weights = [1] * len(U) + W
        
        pb_clauses = PBEnc.leq(lits=variables, weights=weights, bound=UB, top_id=var)
        
        if pb_clauses.nv > var:
            var = pb_clauses.nv + 1
        
        for clause in pb_clauses.clauses:
            clauses.append(clause)
            solver.add_clause(clause)
    
    return clauses, soft_clauses, var, U, solver

def save_solution_to_log(solution, best_value, instance_name, status=""):
    """Save solution to a date-based folder in log directory with instance-specific structure"""
    if solution is None:
        return None
    
    # Create date-based folder name
    current_date = datetime.now()
    date_folder = f"{current_date.day:02d}-{current_date.month:02d}-{current_date.year}"
    
    # Create instance-specific folder structure: log/date_folder/instance_name/n_m_c/
    instance_folder = os.path.join("log", date_folder, instance_name, f"{n}_{m}_{c}")
    
    # Create directory if it doesn't exist
    os.makedirs(instance_folder, exist_ok=True)
    
    # Generate solution data
    x = [[solution[j*m+i] for i in range(m)] for j in range(n)]
    a = [[solution[m*n + j*c + i] for i in range(c)] for j in range(n)]
    cnt = m*n + c*n 
    s = []
    for j in range(n):
        tmp = []
        for i in range(c - time_list[j] + 1):
            tmp.append(solution[cnt])
            cnt += 1
        s.append(tmp)
    
    table = [[0 for t in range(c)] for k in range(m)]
    for k in range(m):
        for t in range(c):
            for j in range(n):
                if x[j][k] > 0 and a[j][t] > 0 and table[k][t] == 0:
                    for l in range(max(0,t-time_list[j]),t+1):
                        if l < len(s[j]) and s[j][l] > 0:
                            table[k][t] = j+1

    # Generate task power list
    task_power_list = ""
    for j in range(n):
        task_power_list += f"task {j+1}: {W[j]}"
        if j < n-1:
            task_power_list += ", "
    
    # Generate task activity list with time ranges and machine assignments
    task_activities = []
    for j in range(n):
        # Find which machine this task is assigned to
        assigned_machine = -1
        for k in range(m):
            if x[j][k] > 0:
                assigned_machine = k + 1
                break
        
        if assigned_machine > 0:
            # Find the start and end times for this task
            start_time = -1
            end_time = -1
            for t in range(c):
                if a[j][t] > 0:
                    if start_time == -1:
                        start_time = t + 1  # Convert to 1-based indexing
                    end_time = t + 1  # Convert to 1-based indexing
                elif end_time > 0:
                    break
            
            if start_time > 0 and end_time > 0:
                task_activities.append(f"task {j+1} [{start_time};{end_time}] in machine {assigned_machine}")
    
    task_activities_text = ", ".join(task_activities)

    # Generate HTML content
    timestamp = current_date.strftime("%Y-%m-%d %H:%M:%S")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Task Assignment - {instance_name} ({n}_{m}_{c})</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            table, th, td {{
                border: 1px solid black;
            }}
            th, td {{
                padding: 8px;
                text-align: center;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .info {{
                margin: 10px 0;
                line-height: 1.6;
            }}
            .best-value {{
                font-size: 18px;
                font-weight: bold;
                color: #2e7d32;
            }}
            .folder-path {{
                font-size: 12px;
                color: #666;
                margin-bottom: 10px;
            }}
            .detailed-info {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                border-left: 4px solid #2e7d32;
            }}
            .section-title {{
                font-weight: bold;
                color: #1976d2;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <h2>Task Assignment - {instance_name}</h2>
        <div class="folder-path">
            <strong>Folder Structure:</strong> log/{date_folder}/{instance_name}/{n}_{m}_{c}/
        </div>
        
        <div class="detailed-info">
            <div class="section-title">Instance Details:</div>
            {instance_name}<br>
            n = {n}, m = {m}, c = {c}<br>
            
            <div class="section-title">Task Power Values:</div>
            {task_power_list}<br>
            
            <div class="section-title">Task Activities:</div>
            {task_activities_text}<br>
        </div>
        
        <div class="info">
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Status:</strong> {status}</p>
            <p class="best-value">Best Value: {best_value}</p>
        </div>
        
        <h3>Task Assignment Table</h3>
        <table>
            <tr>
                <th>Machine</th>
                """ + "".join([f"<th>Time {t+1}</th>" for t in range(c)]) + """
            </tr>
    """

    for k in range(m):
        row = f"<tr><td>Machine {k+1}</td>" + "".join([f"<td>{table[k][t]}</td>" if table[k][t] > 0 else "<td></td>" for t in range(c)]) + "</tr>"
        html_content += row

    html_content += """
        </table>
    </body>
    </html>
    """

    # Save HTML file (without hour in filename, will replace if exists)
    html_filename = f"{instance_name}_bestvalue_{best_value}_{status}.html"
    html_filepath = os.path.join(instance_folder, html_filename)
    
    with open(html_filepath, "w") as file:
        file.write(html_content)
    
    # Also save raw solution data as JSON (without hour in filename, will replace if exists)
    solution_data = {
        'instance': instance_name,
        'configuration': f"{n}_{m}_{c}",
        'timestamp': timestamp,
        'best_value': best_value,
        'status': status,
        'parameters': {'n': n, 'm': m, 'c': c},
        'folder_structure': f"log/{date_folder}/{instance_name}/{n}_{m}_{c}/",
        'assignment_table': table,
        'x_variables': x,
        'a_variables': a,
        's_variables': s
    }
    
    json_filename = f"{instance_name}_bestvalue_{best_value}_{status}.json"
    json_filepath = os.path.join(instance_folder, json_filename)
    
    with open(json_filepath, "w") as file:
        json.dump(solution_data, file, indent=2)
    
    print(f"Solution saved to: {html_filepath}")
    print(f"Data saved to: {json_filepath}")
    print(f"Folder structure: log/{date_folder}/{instance_name}/{n}_{m}_{c}/")
    
    return html_filepath

def print_solution(solution):
    if solution is None:
        # print("No solution found.")
        return None
    else:
        x = [[solution[j*m+i] for i in range(m)] for j in range(n)]
        a = [[solution[m*n + j*c + i] for i in range(c)] for j in range(n)]
        # s = [[solution[m*n + c*n + j*c + i] for i in range(c)] for j in range(n)]
        cnt = m*n + c*n 
        s = []
        for j in range(n):
            tmp = []
            for i in range(c - time_list[j] + 1):
                tmp.append(solution[cnt])
                cnt += 1
            s.append(tmp)
        table = [[0 for t in range(c)] for k in range(m)]
        for k in range(m):
            for t in range(c):
                for j in range(n):
                    if x[j][k] > 0 and a[j][t] > 0 and table[k][t] == 0:
                        for l in range(max(0,t-time_list[j]),t+1):
                            if l < len(s[j]) and s[j][l] > 0:
                                table[k][t] = j+1

        # Generate HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Task Assignment to Machines</title>
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                table, th, td {
                    border: 1px solid black;
                }
                th, td {
                    padding: 8px;
                    text-align: center;
                }
                th {
                    background-color: #f2f2f2;
                }
            </style>
        </head>
        <body>
            <h2>Task Assignment to Machines</h2>
            <table>
                <tr>
                    <th>Machine</th>
                    """ + "".join([f"<th>Time {t+1}</th>" for t in range(c)]) + """
                </tr>
        """

        for k in range(m):
            row = f"<tr><td>Machine {k+1}</td>" + "".join([f"<td>{table[k][t]}</td>" if table[k][t] > 0 else "<td></td>" for t in range(c)]) + "</tr>"
            html_content += row

        html_content += """
            </table>
        </body>
        </html>
        """

        # Write HTML content to a file
        file_path = "task_assignment.html"
        with open(file_path, "w") as file:
            file.write(html_content)

        # Open the HTML file in the default web browser
        # webbrowser.open(file_path)

def optimal(X, S, A, n, m, c, sol, solbb, start_time):
    """Main optimization function"""
    global filename
    
    ip1, ip2 = preprocess(n, m, c, time_list, adj)
    clauses = generate_clauses(n, m, c, time_list, adj, ip1, ip2, X, S, A)
    
    solver = Cadical195()
    for clause in clauses:
        solver.add_clause(clause)
    
    # Check timeout before initial solve
    current_time = time.time()
    remaining_time = 3600 - (current_time - start_time)
    if remaining_time <= 0:
        print("Instance timeout before initial solve")
        return 0, var_counter, clauses, [], "TIMEOUT"
    
    model = solve(solver)
    if model is None:
        print("Initial solve timed out or no solution")
        return 0, var_counter, clauses, [], "TIMEOUT"
    
    best_solution = model
    infinity = 1000000
    best_value, lowval, station = get_value(model, infinity)
    lowval = max(W)
    
    print(f"Initial value: {best_value}")
    print(f"Initial station: {station}")
    
    start_var = var_counter
    clauses, soft_clauses, var, U, solver1 = generate_inagural(
        n, m, c, X, S, A, W, best_value, lowval, clauses, start_var, solver
    )
    
    # Incremental optimization
    while True:
        current_time = time.time()
        if current_time - start_time >= 3600:
            print("Time limit exceeded.")
            instance_name = filename.split(".")[0] if filename else "Unknown"
            save_solution_to_log(best_solution, best_value, instance_name, "Time_Limit_Exceeded")
            return best_value, var, clauses, soft_clauses, "Time Limit Exceeded"
        
        remaining_time = 3600 - (current_time - start_time)
        if remaining_time <= 1:
            print("Time limit exceeded - insufficient time remaining")
            instance_name = filename.split(".")[0] if filename else "Unknown"
            save_solution_to_log(best_solution, best_value, instance_name, "Time_Limit_Exceeded")
            return best_value, var, clauses, soft_clauses, "Time Limit Exceeded"
        
        model = solve(solver1)
        if model is None:
            print("No solution found maxsat or timeout.")
            instance_name = filename.split(".")[0] if filename else "Unknown"
            save_solution_to_log(best_solution, best_value, instance_name, "Optimal")
            return best_value, var, clauses, soft_clauses, "Optimal"
        
        best_solution = model
        _, best_value = get_value2(n, m, c, model, W)
        print(f"Best value: {best_value}", end="\r")
        
        idx = best_value - lowval - 1
        solver1.add_clause([-U[idx-1]])

def write_results_to_csv(ins, n, m, c, val, s_cons, h_cons, peak, status, time_elapsed):
    """Write results to CSV file"""
    global best_result
    
    result = {
        'Instance': ins, 'n': n, 'm': m, 'c': c,
        'Variables': val, 'SoftClauses': s_cons, 'HardClauses': h_cons,
        'OptimalValue': peak, 'Status': status, 'Runtime': time_elapsed
    }
    
    best_result = result.copy()
    
    filename = "Output/Incremental_cadical_2.csv"
    with open(filename, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ins, str(n), str(m), str(c), str(val), 
                        str(s_cons), str(h_cons), str(peak), status, str(time_elapsed)])

def run_single_instance(instance_id):
    """Run optimization for a single instance"""
    global start_time_global, best_result, n, m, c, val, filename
    
    if instance_id >= len(INSTANCES):
        print(f"Invalid instance ID: {instance_id}. Max ID is {len(INSTANCES)-1}")
        return
    
    print(f"\nProcessing instance {instance_id}: {INSTANCES[instance_id][0]}")
    
    start_time_global = time.time()
    reset_globals(instance_id)
    read_input()
    
    X, A, S = generate_variables(n, m, c)
    val = max(S[-1]) if S else 0
    
    start_time = time.time()
    solval, vari, clauses, soft_clauses, status = optimal(X, S, A, n, m, c, 0, 0, start_time)
    end_time = time.time()
    
    write_results_to_csv(filename.split(".")[0], n, m, c, 
                        vari, len(soft_clauses), len(clauses), solval, status, end_time - start_time)
    
    if best_result:
        with open(f'results_incremental_cadical_{instance_id}.json', 'w') as f:
            json.dump(best_result, f)
    
    print(f"Instance {instance_id} completed - Runtime: {end_time - start_time:.2f}s, Status: {status}")

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("Usage:")
        print("  python3 incremental_cadical_2.py              # Run all instances with runlim")
        print("  python3 incremental_cadical_2.py <id>         # Run single instance by ID")
        print("  python3 incremental_cadical_2.py easy         # Run only easy instances (0-38)")
        print("  python3 incremental_cadical_2.py hard         # Run only hard instances (39+)")
        print("  python3 incremental_cadical_2.py all          # Run all instances")
        sys.exit(0)
    
    global INSTANCES
    
    # Handle difficulty selection
    if len(sys.argv) > 1 and sys.argv[1] in ['easy', 'hard', 'all']:
        difficulty = sys.argv[1]
        if difficulty == 'easy':
            INSTANCES = INSTANCES[:39] if len(INSTANCES) >= 39 else INSTANCES
            print(f"Running EASY instances only: {len(INSTANCES)} instances")
        elif difficulty == 'hard':
            INSTANCES = INSTANCES[39:] if len(INSTANCES) > 39 else []
            print(f"Running HARD instances only: {len(INSTANCES)} instances")
        else:
            print(f"Running ALL instances: {len(INSTANCES)} instances")
        sys.argv = [sys.argv[0]]
    
    # Controller mode
    if len(sys.argv) == 1:
        os.makedirs('Output', exist_ok=True)
        excel_file = 'Output/Incremental_cadical_2.xlsx'
        TIMEOUT = 3601
        
        print(f"Running {len(INSTANCES)} instances with {TIMEOUT}s timeout each")
        
        for instance_id in range(len(INSTANCES)):
            instance_name = INSTANCES[instance_id][0]
            print(f"\n{'=' * 50}")
            print(f"Running instance {instance_id}: {instance_name} (m={INSTANCES[instance_id][1]}, c={INSTANCES[instance_id][2]})")
            print(f"{'=' * 50}")
            
            # Clean up previous result files
            for temp_file in [f'results_incremental_cadical_{instance_id}.json', 
                             f'checkpoint_incremental_cadical_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            command = f"./runlim -r {TIMEOUT} python3 incremental_cadical_2.py {instance_id}"
            
            try:
                process = subprocess.Popen(command, shell=True)
                process.wait()
                time.sleep(1)
                
                # Process results
                if os.path.exists(f'results_incremental_cadical_{instance_id}.json'):
                    with open(f'results_incremental_cadical_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                    
                    print(f"Instance {instance_name} - Status: {result['Status']}")
                    print(f"Optimal Value: {result['OptimalValue']}, Runtime: {result['Runtime']}")
                    
                    # Save to Excel
                    if os.path.exists(excel_file):
                        try:
                            existing_df = pd.read_excel(excel_file)
                            result_df = pd.DataFrame([result])
                            existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                        except:
                            existing_df = pd.DataFrame([result])
                    else:
                        existing_df = pd.DataFrame([result])
                    
                    existing_df.to_excel(excel_file, index=False)
                else:
                    print(f"No results found for instance {instance_name}")
                    
            except Exception as e:
                print(f"Error running instance {instance_name}: {str(e)}")
            
            # Clean up temp files
            for temp_file in [f'results_incremental_cadical_{instance_id}.json', 
                             f'checkpoint_incremental_cadical_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        print(f"\nAll instances completed. Results saved to {excel_file}")
    
    # Single instance mode
    else:
        instance_id = int(sys.argv[1])
        run_single_instance(instance_id)
    ["BUXEY", 10, 36],      # 46
    # Hard/BUXEY count: 7

    # SAWYER
    ["SAWYER", 14, 25],     # 47
    ["SAWYER", 7, 47],      # 48
    ["SAWYER", 8, 41],      # 49
    ["SAWYER", 12, 30],     # 50
    ["SAWYER", 13, 27],     # 51
    ["SAWYER", 11, 33],     # 52
    ["SAWYER", 10, 36],     # 53 ???
    ["SAWYER", 7, 54],      # 54
    ["SAWYER", 5, 75],      # 55
    # Hard/SAWYER count: 9

    # GUNTHER
    ["GUNTHER", 9, 54],     # 57
    ["GUNTHER", 9, 61],     # 58
    ["GUNTHER", 14, 41],    # 59
    ["GUNTHER", 12, 44],    # 60
    ["GUNTHER", 11, 49],    # 61
    ["GUNTHER", 8, 69],     # 62
    ["GUNTHER", 7, 81],     # 63
    # Hard/GUNTHER count: 7

    # WARNECKE
    ["WARNECKE", 25, 65],   # 64
    ["WARNECKE", 31, 54],   # 65
    ["WARNECKE", 29, 56],   # 66
    ["WARNECKE", 29, 58],   # 67 
    ["WARNECKE", 27, 60],   # 68
    ["WARNECKE", 27, 62],   # 69
    ["WARNECKE", 24, 68],   # 70
    ["WARNECKE", 23, 71],   # 71
    ["WARNECKE", 22, 74],   # 72
    ["WARNECKE", 21, 78],   # 73
    ["WARNECKE", 20, 82],   # 74
    ["WARNECKE", 19, 86],   # 75
    ["WARNECKE", 17, 92],   # 76
    ["WARNECKE", 17, 97],   # 77
    ["WARNECKE", 15, 104],  # 78
    ["WARNECKE", 14, 111],  # 79
    # Hard/WARNECKE count: 16

    # Lutz2
    ["LUTZ2", 49, 11],      # 80
    ["LUTZ2", 44, 12],      # 81
    ["LUTZ2", 40, 13],      # 82
    ["LUTZ2", 37, 14],      # 83
    ["LUTZ2", 34, 15],      # 84
    ["LUTZ2", 31, 16],      # 85
    ["LUTZ2", 29, 17],      # 86
    ["LUTZ2", 28, 18],      # 87
    ["LUTZ2", 26, 19],      # 88
    ["LUTZ2", 25, 20],      # 89
    ["LUTZ2", 24, 21],      # 90
    # Hard/Lutz2 count: 11

    # Hard families total count: 50

    # Total: 89
]

# Override instances if timeout file exists
if os.path.exists("incremental_cadical_timeout.txt"):
    with open("incremental_cadical_timeout.txt", "r") as f:
        timeout_instances = [line.strip() for line in f if line.strip()]
    # Filter file_name1 to only include instances in timeout file
    file_name1 = [instance for instance in file_name1 if instance[0] in timeout_instances]

def reset(idx):
    global n, m, c, val, cons, sol, solbb, type, filename, W, neighbors, reversed_neighbors, visited, toposort, clauses, time_list, adj, forward, var_map, var_counter, current_instance_id
    current_instance_id = idx
    m = file_name1[idx][1]
    c = file_name1[idx][2]
    val = 0
    cons = 0
    sol = 0
    solbb = 0
    type = 1
    var_counter = 0
    var_map = {}
    filename = file_name1[idx][0] + ".IN2"
    W = [int(line.strip()) for line in open('task_power/'+file_name1[idx][0]+'.txt')]
    neighbors = [[ 0 for i in range(200)] for j in range(200)]
    reversed_neighbors = [[ 0 for i in range(200)] for j in range(200)]
    visited = [False for i in range(200)]
    toposort = []
    clauses = []
    time_list = []
    adj = []
    forward = [0 for i in range(200)]

def run_single_instance(instance_id):
    """Run a single instance by ID"""
    global start_time_global, best_result, n, m, c, val, cons, sol, solbb, type, filename, W, neighbors, reversed_neighbors, visited, toposort, clauses, time_list, adj, forward, var_map, var_counter
    
    if instance_id >= len(file_name1):
        print(f"Invalid instance ID: {instance_id}. Max ID is {len(file_name1)-1}")
        return
    
    print(f"\nProcessing instance {instance_id}: {file_name1[instance_id][0]}")
    
    start_time_global = time.time()
    reset(instance_id)
    read_input()
    X, A, S = generate_variables(n,m,c)
    val = max(S)

    var_counter = max(val)
    var_map = {}

    sol = 0
    solbb = 0
    start_time = time.time()
    solval, vari, clauses, soft_clauses, status = optimal(X,S,A,n,m,c,sol,solbb,start_time)
    end_time = time.time()
    
    # Write results
    write_fancy_table_to_csv(filename.split(".")[0], n, m, c, vari, len(soft_clauses), len(clauses), solval, status, end_time - start_time)
    
    # Save JSON result for controller
    if best_result:
        with open(f'results_incremental_cadical_{instance_id}.json', 'w') as f:
            json.dump(best_result, f)
    
    print(f"Instance {instance_id} completed - Runtime: {end_time - start_time:.2f}s, Status: {status}")

def main():
    """Original main function - runs all instances sequentially"""
    global n, m, c, val, cons, sol, solbb, type, filename, W, neighbors, reversed_neighbors, visited, toposort, clauses, time_list, adj, forward, var_map, var_counter, start_time_global
    
    start_time_global = time.time()
    # Run all 89 instances (change to 39 for easy instances only)
    for idx in range(0, len(file_name1)):
        reset(idx)
        read_input()
        X, A, S = generate_variables(n,m,c)
        val = max(S)

        # print(val)
        var_counter = max(val)
        var_map = {}

        sol = 0
        solbb = 0
        start_time = time.time()
        solval, vari, clauses, soft_clauses, status = optimal(X,S,A,n,m,c,sol,solbb,start_time)
        end_time = time.time()
        write_fancy_table_to_csv(filename.split(".")[0], n, m, c, vari, len(soft_clauses), len(clauses), solval, status, end_time - start_time)

if __name__ == "__main__":
    # Help message
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("Usage:")
        print("  python3 incremental_cadical_2.py              # Run all instances with runlim")
        print("  python3 incremental_cadical_2.py <id>         # Run single instance by ID")
        print("  python3 incremental_cadical_2.py easy         # Run only easy instances (0-38)")
        print("  python3 incremental_cadical_2.py hard         # Run only hard instances (39+)")
        print("  python3 incremental_cadical_2.py all          # Run all instances")
        print("")
        print(f"Available instances: {len(file_name1)} total")
        print("Easy instances: 0-38 (39 instances)")
        print("Hard instances: 39+ (remaining instances)")
        print("")
        print("Files created:")
        print("  Output/Incremental_cadical_all.csv   # CSV results")
        print("  Output/Incremental_cadical_all.xlsx  # Excel results")
        print("  results_incremental_cadical_<id>.json # Individual results")
        print("")
        print("Optional timeout file: incremental_cadical_timeout.txt")
        print("  (List instance names to run only those instances)")
        sys.exit(0)
    
    # Check for special arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['easy', 'hard', 'all']:
        difficulty = sys.argv[1]
        if difficulty == 'easy':
            # Only run easy instances (first 39)
            instances_to_run = file_name1[:39] if len(file_name1) >= 39 else file_name1
            print(f"Running EASY instances only: {len(instances_to_run)} instances")
        elif difficulty == 'hard':
            # Only run hard instances (40-89)
            instances_to_run = file_name1[39:] if len(file_name1) > 39 else []
            print(f"Running HARD instances only: {len(instances_to_run)} instances")
        else:  # 'all'
            instances_to_run = file_name1
            print(f"Running ALL instances: {len(instances_to_run)} instances")
        
        # Run selected instances with controller mode
        file_name1 = instances_to_run
        sys.argv = [sys.argv[0]]  # Reset argv to trigger controller mode
    
    # Controller mode - run all instances with runlim
    if len(sys.argv) == 1:
        # Create Output folder if it doesn't exist
        if not os.path.exists('Output'):
            os.makedirs('Output')
        
        # Read existing Excel file to check completed instances
        excel_file = 'Output/Incremental_cadical_2.xlsx'
        csv_file = 'Output/Incremental_cadical_2.csv'
        
        completed_instances = []

        
        # Set timeout (1 hour = 3600s)
        TIMEOUT = 3601
        
        print(f"Running {len(file_name1)} instances with {TIMEOUT}s timeout each")
        
        # Run all instances with runlim
        # Here
        for instance_id in range(0, len(file_name1)):
            instance_name = file_name1[instance_id][0]
            
            print(f"\n{'=' * 50}")
            print(f"Running instance {instance_id}: {instance_name} (m={file_name1[instance_id][1]}, c={file_name1[instance_id][2]})")
            print(f"{'=' * 50}")
            
            # Clean up previous result files
            for temp_file in [f'results_incremental_cadical_{instance_id}.json', 
                              f'checkpoint_incremental_cadical_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Run instance with timeout
            command = f"./runlim -r {TIMEOUT} python3 incremental_cadical_2.py {instance_id}"
            
            try:
                process = subprocess.Popen(command, shell=True)
                process.wait()
                time.sleep(1)
                
                # Check results
                result = None
                
                if os.path.exists(f'results_incremental_cadical_{instance_id}.json'):
                    with open(f'results_incremental_cadical_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                
                # Process results
                if result:
                    print(f"Instance {instance_name} - Status: {result['Status']}")
                    print(f"Optimal Value: {result['OptimalValue']}, Runtime: {result['Runtime']}")
                    
                    # Convert JSON to Excel format
                    if os.path.exists(excel_file):
                        try:
                            existing_df = pd.read_excel(excel_file)
                            result_df = pd.DataFrame([result])
                            existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                        except:
                            existing_df = pd.DataFrame([result])
                    else:
                        existing_df = pd.DataFrame([result])
                    
                    existing_df.to_excel(excel_file, index=False)
                    print(f"Results saved to {excel_file}")
                else:
                    print(f"No results found for instance {instance_name}")
                    
            except Exception as e:
                print(f"Error running instance {instance_name}: {str(e)}")
            
            # Clean up temp files
            for temp_file in [f'results_incremental_cadical_{instance_id}.json', 
                              f'checkpoint_incremental_cadical_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        print(f"\nAll instances completed. Results saved to {excel_file}")
    
    # Single instance mode
    else:
        instance_id = int(sys.argv[1])
        run_single_instance(instance_id)

