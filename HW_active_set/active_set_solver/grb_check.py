import gurobipy as gp
from gurobipy import GRB

try:
    # Create a new model
    model = gp.Model("qp")

    # Add decision variables (nonnegative by default)
    x1 = model.addVar(lb=0, name="x1")
    x2 = model.addVar(lb=0, name="x2")

    # Set the objective:
    #   minimize x1^2 + 2x2^2 - 2x1 - 6x2 - 2x1*x2
    # Gurobi handles quadratic expressions directly.
    model.setObjective(x1 * x1 + 2 * x2 * x2 - 2 * x1 - 6 * x2 - 2 * x1 * x2, GRB.MINIMIZE)

    # Add constraints:
    # 0.5*x1 + 0.5*x2 <= 1
    model.addConstr(0.5 * x1 + 0.5 * x2 <= 1, "c0")
    # -x1 + 2*x2 <= 2
    model.addConstr(-x1 + 2 * x2 <= 2, "c1")

    # Optimize the model
    model.optimize()

    # Print the solution
    if model.status == GRB.OPTIMAL:
        for v in model.getVars():
            print(f"{v.varName} = {v.x:.4f}")
        print(f"Optimal objective value: {model.objVal:.4f}")
    else:
        print("No optimal solution found.")

except gp.GurobiError as e:
    print("Gurobi Error:", e)
except Exception as e:
    print("Error:", e)