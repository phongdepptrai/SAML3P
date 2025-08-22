from math import inf
import re
import time

from pysat.solvers import Cadical195 as Glucose4
from tabulate import tabulate
import webbrowser
import sys
import csv
import signal
# input variables in database ?? mertens 1
n = 35
m = 14
c = 40
val = 0
cons = 0
sol = 0
solbb = 0
type = 1
#           0              1                2           3           4           5           6           7               8                   9
file = ["MITCHELL.IN2","MERTENS.IN2","BOWMAN.IN2","ROSZIEG.IN2","BUXEY.IN2","HESKIA.IN2","SAWYER.IN2","JAESCHKE.IN2","MANSOOR.IN2",
        "JACKSON.IN2","GUNTHER.IN2", "WARNECKE.IN2"]
#            9          10              11          12          13          14          15          16          17   
filename = file[3]

fileName = filename.split(".")

with open('task_power/'+fileName[0]+'.txt', 'r') as file:
    W = [int(line.strip()) for line in file]

neighbors = [[ 0 for i in range(n)] for j in range(n)]
reversed_neighbors = [[ 0 for i in range(n)] for j in range(n)]

visited = [False for i in range(n)]
toposort = []
clauses = []
time_list = []
adj = []
var_map = {}
var_counter = 0
# W = [41, 13, 21, 24, 11, 11, 41, 32, 31, 25, 29, 25, 31, 3, 14, 37, 34, 6, 18, 35, 18, 19, 25, 40, 20, 20, 36, 23, 29, 48, 41, 20, 31, 25, 1]

file_name = [
    ["MERTENS", 6, 6],      #0
    ["MERTENS", 2, 18],     #1
    ["BOWMAN", 5, 20],      #2
    ["JAESCHKE", 8, 6],     #3
    ["JAESCHKE", 3, 18],    #4
    ["JACKSON", 8, 7],      #5
    ["JACKSON", 3, 21],     #6
    ["MANSOOR", 4, 48],     #7
    ["MANSOOR", 2, 94],     #8
    ["MITCHELL", 8, 14],    #9
    ["MITCHELL", 3, 39],    #10
    ["ROSZIEG", 10, 14],    #11
    ["ROSZIEG", 4, 32],     #12
    ["ROSZIEG", 6, 25],     #13
    ["HESKIA", 8, 138],     #14
    ["HESKIA", 3, 342],     #15
    ["HESKIA", 5, 205],     #16
    ["BUXEY", 14, 25],      #17
    ["BUXEY", 7, 47],       #18
    ["BUXEY", 8, 41],       #19
    ["BUXEY", 11, 33],      #20
    ["SAWYER", 14, 25],     #21
    ["SAWYER", 7, 47],      #22
    ["SAWYER", 8, 41],      #23
    ["SAWYER", 12, 30],     #24
    ["GUNTHER", 14, 40],    #25
    ["GUNTHER", 9, 54],     #26
    ["GUNTHER", 9, 61],     #27
    ["WARNECKE",25, 65]     #28
    ]




def read_input():
    global n, m, c, filename, W, neighbors, reversed_neighbors, visited, toposort, clauses, time_list, adj
    cnt = 0
    # Use a context-managed file read to avoid fileinput's global state and ensure proper closure
    with open('presedent_graph/' + filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                if cnt == 0:
                    n = int(line)
                elif cnt <= n:  # type: ignore
                    time_list.append(int(line))
                else:
                    parts = line.split(",")
                    if(parts[0] != "-1" and parts[1] != "-1"):
                        adj.append([int(parts[0]) - 1, int(parts[1]) - 1])
                        neighbors[int(parts[0]) - 1][int(parts[1]) - 1] = 1
                        reversed_neighbors[int(parts[1]) - 1][int(parts[0]) - 1] = 1
                    else:
                        break
                cnt = cnt + 1

def generate_variables(n,m,c):
    return [[j*m+i+1 for i in range (m)] for j in range(n)], [[m*n + j*c + i + 1 for i in range (c)] for j in range(n)], [[m*n + c*n + j*c + i + 1 for i in range (c)] for j in range(n)]

def dfs(v):
    visited[v] = True
    for i in range(n):
        if(neighbors[v][i] == 1 and visited[i] == False):
            dfs(i)
    toposort.append(v)
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
            #?????
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
    # 1
    for j in range (0, n):
        constraint = []
        for k in range (0, m):
            if ip1[j][k] == 1:
                continue
            constraint.append(X[j][k])
        clauses.append(constraint)
    # 2 
    for j in range(0,n):
        for k1 in range (0,m-1):
            for k2 in range(k1+1,m):
                if ip1[j][k1] == 1 or ip1[j][k2] == 1:
                    continue
                clauses.append([-X[j][k1], -X[j][k2]])

    #3
    for a,b in adj:
        for k in range (0, m-1):
            for h in range(k+1, m):
                if ip1[b][k] == 1 or ip1[a][h] == 1:
                    continue
                clauses.append([-X[b][k], -X[a][h]])
    print("first 3 constraints:", len(clauses))

    #4

    for j in range(n):
        clauses.append([S[j][t] for t in range (c-time_list[j]+1)])

    #5
    for j in range(n):
        for k in range(c-time_list[j]):
            for h in range(k+1, c-time_list[j]+1):
                clauses.append([-S[j][k], -S[j][h]])

    #6
    for j in range(n):
        for t in range(c-time_list[j]+1,c):
            if t > c- time_list[j]:
                clauses.append([-S[j][t]])
    print("4 5 6 constraints:", len(clauses))

    #7
    for i in range(n-1):
        for j in range(i+1,n):
            for k in range (m):
                if ip1[i][k] == 1 or ip1[j][k] == 1 :
                    continue
                for t in range(c):
                    # if ip2[i][k][t] == 1 or ip2[j][k][t] == 1:
                    #     continue
                    clauses.append([-X[i][k], -X[j][k], -A[i][t], -A[j][t]])
    print("7 constraints:", len(clauses))
    #8
    for j in range(n):
        for t in range (c-time_list[j]+1):
            for l in range (time_list[j]):
                if(time_list[j] >= c/2 and t+l >= c-time_list[j] and t+l < time_list[j]):
                    continue
                clauses.append([-S[j][t],A[j][t+l]])
    
    print("8 constraints:", len(clauses))

    # addtional constraints
    # a task cant run before its active time

    # for j in range(n):
    #     for t in range (c-time_list[j]+1):
    #         for l in range (t):
    #             if(time_list[j] >= c/2 and l >= c-time_list[j] and l < time_list[j]):
    #                 continue
    #             clauses.append([-S[j][t],-A[j][l]])


    # addtional constraints option 2


    # for j in range(n):
    #     for t in range (c-1): 
    #         if(time_list[j] >= c/2 and t+1 >= c-time_list[j] and t+1 < time_list[j]):
    #             continue
    #         clauses.append([ -A[j][t], A[j][t+1] , S[j][max(0,t-time_list[j]+1)]])
    
    #9
    for i, j in adj:
        for k in range(m):
            if ip1[i][k] == 1 or ip1[j][k] == 1:
                continue
            for t1 in range(c - time_list[i] +1):
                #t1>t2
                for t2 in range(c-time_list[j]+1):
                    if ip2[i][k][t1] == 1 or ip2[j][k][t2] == 1:
                        continue
                    if t1 > t2:
                        clauses.append([-X[i][k], -X[j][k], -S[i][t1], -S[j][t2]])
    cons = len(clauses)
    print("Constraints:",cons)

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
    if solver.solve():
        model = solver.get_model()
        return model
    else:
        # print("no solution")
        return None

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Solver timeout")

def solve_with_timeout(solver, timeout_seconds):
    try:
        # Set up timeout signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        # Try to solve
        result = solve(solver)
        
        # Cancel timeout if we finish early
        signal.alarm(0)
        return result
        
    except TimeoutException:
        signal.alarm(0)  # Cancel timeout
        print(f"Solver timed out after {timeout_seconds} seconds")
        return None

def print_solution(solution):
    if solution is None:
        # print("No solution found.")
        return None
    else:
        x = [[solution[j*m+i] for i in range(m)] for j in range(n)]
        s = [[solution[m*n + j*c + i] for i in range(c)] for j in range(n)]
        a = [[solution[m*n + c*n + j*c + i] for i in range(c)] for j in range(n)]
        table = [[0 for t in range(c)] for k in range(m)]
        for k in range(m):
            for t in range(c):
                for j in range(n):
                    if x[j][k] > 0 and a[j][t] > 0 and table[k][t] == 0:
                        for l in range(max(0,t-time_list[j]),t+1):
                            if s[j][l] > 0:
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

def get_value(solution,best_value):
    if solution is None:
        return 100, []
    else:
        x = [[  solution[j*m+i] for i in range (m)] for j in range(n)]
        s = [[  solution[m*n + j*c + i ] for i in range (c)] for j in range(n)]
        a = [[  solution[m*n + c*n + j*c + i ] for i in range (c)] for j in range(n)]
        t = 0
        value = 0

        for t in range(c):
            tmp = 0
            for j in range(n):
                if a[j][t] > 0 :
                    # tmp = tmp + W[j]
                    for l in range(max(0,t-time_list[j]),t+1):
                        if s[j][l] > 0:
                            tmp = tmp + W[j]
                            # print(tmp)
                            break
                
            if tmp > value:
                value = tmp
                # print(value)

        constraints = []
        for t in range(c):
            tmp = 0
            station = []
            for j in range(n):
                if a[j][t] > 0:
                    # tmp = tmp + W[j]
                    # station.append(j+1)
                    for l in range(max(0,t-time_list[j]),t+1):
                        if s[j][l] > 0:
                            tmp = tmp + W[j]
                            station.append(j+1)
                            break
            if tmp >= min(best_value, value):
                constraints.append(station)
                # print("value:",value)
        unique_constraints = list(map(list, set(map(tuple, constraints))))

        return value, unique_constraints

def write_fancy_table_to_csv(ins, n, m, c, var_count, cons_count, peak, sol_count, solbb_count, status, time_taken, filename="SAML3P_cad_results.csv"):
    with open("Output/" + filename, "a", newline='') as f:
        writer = csv.writer(f)
        row = []
        row.append(ins)
        row.append(str(n))
        row.append(str(m))
        row.append(str(c))
        row.append(str(var_count))
        row.append(str(cons_count))
        row.append(str(peak))
        row.append(str(sol_count))
        row.append(str(solbb_count))
        row.append(status)
        row.append(str(time_taken))
        writer.writerow(row)

def reset(idx):
    global n, m, c, val, cons, sol, solbb, type, filename, W, neighbors, reversed_neighbors, visited, toposort, clauses, time_list, adj, var_map, var_counter
    m = file_name[idx][1]
    c = file_name[idx][2]
    val = 0
    cons = 0
    sol = 0
    solbb = 0
    type = 1
    var_counter = 0
    var_map = {}
    filename = file_name[idx][0] + ".IN2"
    W = [int(line.strip()) for line in open('task_power/'+file_name[idx][0]+'.txt')]
    neighbors = [[ 0 for i in range(100)] for j in range(100)]
    reversed_neighbors = [[ 0 for i in range(100)] for j in range(100)]
    visited = [False for i in range(100)]
    toposort = []
    clauses = []
    time_list = []
    adj = []

def optimal(X,S,A,n,m,c,sol,solbb,start_time):
    print(n,m,c)
    ip1,ip2 = preprocess(n,m,c,time_list,adj)

    # print(ip2[])


    clauses = generate_clauses(n,m,c,time_list,adj,ip1,ip2,X,S,A)

    solver = Glucose4()
    for clause in clauses:
        solver.add_clause(clause)

    # Check timeout before initial solve  
    current_time = time.time()
    remaining_time = 3600 - (current_time - start_time)
    if remaining_time <= 0:
        print("Instance timeout before initial solve")
        return None, sol, solbb, float('inf')

    # Use timeout for initial solve
    model = solve_with_timeout(solver, min(int(remaining_time), 3600))
    if model is None:
        print("Initial solve timed out or no solution")
        return None, sol, solbb, float('inf')
        
    bestSolution = model 
    infinity = 1000000
    result = get_value(model, infinity)

    bestValue, station = result
    print("initial value:",bestValue)
    print("initial station:",station)
    for t in range(c):
        for stations in station:
            
            solver.add_clause([-A[j-1][t] for j in stations])
    sol = 1
    solbb = 1
    while True:
        # Check timeout
        current_time = time.time()
        if current_time - start_time >= 3600:
            print("Instance timeout during optimization")
            return bestSolution, sol, solbb, bestValue
            
        remaining_time = 3600 - (current_time - start_time)
        if remaining_time <= 1:  # Need at least 1 second
            print("Instance timeout - insufficient time remaining")
            return bestSolution, sol, solbb, bestValue
            
        sol = sol + 1
        # Use timeout for each iterative solve
        model = solve_with_timeout(solver, min(int(remaining_time), 3600))  # Max 3600s per iteration

        if model is None:
            # Could be timeout or no more solutions
            return bestSolution, sol, solbb, bestValue
            
        value, station = get_value(model, bestValue)
        # print("value:",value)
        # print("station:",station)
        if value < bestValue:
            solbb = sol
            bestSolution = model
            bestValue = value
            # print("new value:",bestValue)
            # print("new station:",station)

        for t in range(c):
            for stations in station:
                solver.add_clause([-A[j-1][t] for j in stations])
                # print(stations)

def main():
    global n, m, c, val, cons, sol, solbb, type, filename, W, neighbors, reversed_neighbors, visited, toposort, clauses, time_list, adj, var_map, var_counter
    
    # Create Output directory if it doesn't exist
    import os
    if not os.path.exists("Output"):
        os.makedirs("Output")
    
    # Write CSV header only if file doesn't exist
    if not os.path.exists("Output/SAML3P_cad_results.csv"):
        with open("Output/SAML3P_cad_results.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Instance", "n", "m", "c", "Variables", "Constraints", "Peak_Power", "Solutions", "Best_Solutions", "Status", "Time"])
    
    for idx in range(0, 29):
        print(f"Processing instance {idx + 1}/29: {file_name[idx][0]}")
        reset(idx)
        read_input()
        X, S, A = generate_variables(n,m,c)
        val = max(max(row) for row in A)

        var_counter = val
        var_map = {}

        sol = 0
        solbb = 0
        start_time = time.time()
        solution, sol, solbb, solval = optimal(X,S,A,n,m,c,sol,solbb,start_time)
        end_time = time.time()
        
        status = "Optimal" if solution is not None else "No Solution"
        peak_power = solval if solution is not None else 0
        
        write_fancy_table_to_csv(
            filename.split(".")[0], 
            n, m, c, 
            var_counter, 
            len(clauses), 
            peak_power, 
            sol, 
            solbb, 
            status, 
            end_time - start_time
        )
        
        print(f"Instance {file_name[idx][0]} completed: Peak Power = {peak_power}, Time = {end_time - start_time:.2f}s")

main()

