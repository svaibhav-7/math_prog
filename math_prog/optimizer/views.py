from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
import io
import urllib, base64
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


def simplex(request):
    if request.method == 'POST':
        try:
            c_x1 = float(request.POST.get('objective_x1'))
            c_x2 = float(request.POST.get('objective_x2'))

            constraints_x1 = list(map(float, request.POST.getlist('constraints_x1[]')))
            constraints_x2 = list(map(float, request.POST.getlist('constraints_x2[]')))
            constraints_rhs = list(map(float, request.POST.getlist('constraints_rhs[]')))

            A = np.column_stack((constraints_x1, constraints_x2))
            b = np.array(constraints_rhs)
            c = np.array([-c_x1, -c_x2])  # Negate for maximization

            result = linprog(c, A_ub=A, b_ub=b, method='highs')

            if result.success:
                return render(request, 'simplex.html', {
                    'solution': result.x,
                    'optimal_value': -result.fun
                })
            else:
                return render(request, 'simplex.html', {'error': 'No feasible solution'})

        except Exception as e:
            return render(request, 'simplex.html', {'error': str(e)})

    return render(request, 'simplex.html')
