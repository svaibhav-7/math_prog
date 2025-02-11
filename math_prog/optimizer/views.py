from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
import numpy as np  
from scipy.optimize import linprog  
from django.shortcuts import render  
from django import template  
import io
import urllib, base64
from scipy.optimize import linprog
from django.shortcuts import render
import numpy as np
from scipy.optimize import linprog

def home(request):
    return render(request, 'home.html')

def graphical(request):
    if request.method == 'POST':
        try:
            objective_type = request.POST.get('objective_type')
            c_x1 = float(request.POST.get('objective_x1'))
            c_x2 = float(request.POST.get('objective_x2'))

            constraints_x1 = list(map(float, request.POST.getlist('constraints_x1[]')))
            constraints_x2 = list(map(float, request.POST.getlist('constraints_x2[]')))
            constraints_operator = request.POST.getlist('constraints_operator[]')
            constraints_rhs = list(map(float, request.POST.getlist('constraints_rhs[]')))

            # Set up the figure
            fig, ax = plt.subplots(figsize=(6, 6))

            # Plot constraints
            x = np.linspace(0, 10, 400)
            feasible_region = np.ones_like(x) * np.inf

            for i in range(len(constraints_x1)):
                a1, a2, b = constraints_x1[i], constraints_x2[i], constraints_rhs[i]
                if constraints_operator[i] == "<=":
                    y = (b - a1 * x) / a2
                    feasible_region = np.minimum(feasible_region, y)
                    ax.fill_between(x, y, 10, alpha=0.2)
                else:
                    y = (b - a1 * x) / a2
                    feasible_region = np.maximum(feasible_region, y)
                    ax.fill_between(x, 0, y, alpha=0.2)

            # Plot objective function line
            y_obj = (-c_x1 * x) / c_x2
            ax.plot(x, y_obj, label='Objective Function', linestyle='--')

            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.legend()
            ax.grid(True)

            # Save plot to a string buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            return render(request, 'graphical.html', {
                'solution': 'See graph',
                'optimal_value': 'Computed from graph',
                'image': string
            })

        except Exception as e:
            return render(request, 'graphical.html', {'error': str(e)})

    return render(request, 'graphical.html')
from django.shortcuts import render
import numpy as np

from django.shortcuts import render
import numpy as np

def simplex(c, A, b):
    """
    Solves the Linear Programming problem using the Simplex Method:
    Maximize: Z = c^T * x
    Subject to: A * x <= b, x >= 0
    """
    num_constraints, num_variables = A.shape

    # Add slack variables to convert inequalities to equalities
    slack_vars = np.eye(num_constraints)
    tableau = np.hstack((A, slack_vars, b.reshape(-1, 1)))

    # Add the objective function row to the tableau
    obj_row = np.hstack((-c, np.zeros(num_constraints + 1)))
    tableau = np.vstack((tableau, obj_row))

    # Total variables: original + slack
    num_total_vars = num_variables + num_constraints

    # Simplex iterations
    while True:
        # Check optimality: if all coefficients in the objective row are >= 0, we're done
        if all(tableau[-1, :-1] >= 0):
            break

        # Determine the entering variable (most negative coefficient)
        pivot_col = np.argmin(tableau[-1, :-1])

        # Determine the leaving variable (smallest positive ratio of RHS / pivot column)
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf  # Ignore non-positive ratios
        pivot_row = np.argmin(ratios)

        if np.all(ratios == np.inf):
            raise ValueError("The problem is unbounded.")

        # Pivot operation
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element

        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Extract the solution for the original variables
    solution = np.zeros(num_total_vars)
    for i in range(num_constraints):
        basic_var_index = np.where(tableau[i, :-1] == 1)[0]
        if len(basic_var_index) == 1 and basic_var_index[0] < num_total_vars:
            solution[basic_var_index[0]] = tableau[i, -1]

    optimal_value = tableau[-1, -1]
    return solution[:num_variables], optimal_value

def simplex_page(request):
    # Fixed dimensions for this example: 3 variables and 3 constraints
    num_vars = 3
    num_cons = 3

    range_vars = range(num_vars)
    range_cons = range(num_cons)

    if request.method == 'POST':
        try:
            # Read objective type ("maximize" or "minimize")
            objective_type = request.POST.get("objective_type", "maximize")
            # Get objective coefficients
            c = [float(request.POST.get(f'c{i}', 0)) for i in range_vars]
            # For minimization, convert to maximization by multiplying coefficients by -1
            if objective_type == "minimize":
                c = [-coef for coef in c]

            # Build constraint matrix A and RHS vector b
            A_list = []
            b_list = []
            for i in range_cons:
                # Read coefficients for constraint i
                row = [float(request.POST.get(f'A_{i}_{j}', 0)) for j in range_vars]
                # Read the inequality sign for constraint i (default is "<=")
                inequality = request.POST.get(f'inequality_{i}', "<=")
                # Read right-hand side value for constraint i
                b_val = float(request.POST.get(f'b{i}', 0))
                # If the inequality is ">=", multiply the row and b by -1 to convert it to "<="
                if inequality == ">=":
                    row = [-coef for coef in row]
                    b_val = -b_val
                elif inequality == "=":
                    # For now, equality constraints are not supported.
                    raise ValueError("Equality constraints are not supported in this implementation.")
                # For "<=", no change is needed.
                A_list.append(row)
                b_list.append(b_val)
            A = np.array(A_list)
            b = np.array(b_list)

            # Solve the simplex problem
            solution, optimal_value = simplex(np.array(c), A, b)
            # If original problem was minimization, adjust the optimal value by multiplying by -1
            if objective_type == "minimize":
                optimal_value = -optimal_value

            result = {
                'solution': solution.tolist(),
                'optimal_value': optimal_value
            }
        except Exception as e:
            result = {'error': str(e)}

        return render(request, 'simplex.html', {
            'result': result,
            'range_vars': range_vars,
            'range_cons': range_cons
        })
    else:
        return render(request, 'simplex.html', {
            'range_vars': range_vars,
            'range_cons': range_cons
        })


register = template.Library()  

@register.filter  
def get_item(dictionary, key):  
    return dictionary[key]  


def solve_transportation_problem(cost_matrix, supply, demand):  
    cost_matrix = np.array(cost_matrix)  
    supply = np.array(supply)  
    demand = np.array(demand)  

    m, n = cost_matrix.shape  
    c = cost_matrix.flatten()  

    A_eq = []  
    b_eq = []  

    # Supply constraints  
    for i in range(m):  
        row_constraint = [0] * (m * n)  
        for j in range(n):  
            row_constraint[i * n + j] = 1  
        A_eq.append(row_constraint)  
        b_eq.append(supply[i])  

    # Demand constraints  
    for j in range(n):  
        col_constraint = [0] * (m * n)  
        for i in range(m):  
            col_constraint[i * n + j] = 1  
        A_eq.append(col_constraint)  
        b_eq.append(demand[j])  

    # Solve the problem using linprog  
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')  

    if result.success:  
        solution_matrix = result.x.reshape(m, n)  
        # Convert numpy array to a list of lists (rows)  
        solution_rows = solution_matrix.tolist()  
        return {  
            "solution": solution_rows,  
            "total_cost": result.fun,  
            "status": result.message,  
        }  
    else:  
        return {  
            "solution": None,  
            "total_cost": None,  
            "status": result.message,  
        }  


# View to handle the transportation problem  
def transportation_page(request):  
    range_data = range(3)  # Adjust this based on matrix size (3x3 here)  
    if request.method == 'POST':  
        cost_matrix = [[int(request.POST[f'cost_{i}_{j}']) for j in range(3)] for i in range(3)]  
        supply = [int(request.POST[f'supply_{i}']) for i in range(3)]  
        demand = [int(request.POST[f'demand_{j}']) for j in range(3)]  

        result = solve_transportation_problem(cost_matrix, supply, demand)  

        return render(request, 'transportation.html', {  
            'result': result,  
            'cost_matrix': cost_matrix,  
            'supply': supply,  
            'demand': demand,  
            'range_data': range_data,  
        })  
    else:  
        return render(request, 'transportation.html', {'range_data': range_data})