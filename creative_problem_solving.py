import random
import time
from typing import Any, Callable, Dict, List


class CreativeProblemSolving:
    """
    Implements creative problem-solving capabilities.
    """

    def __init__(self):
        self.problem_history = []
        self.solution_approaches = {}
        self.evaluation_history = []
        self.creativity_techniques = {}

    def register_creativity_technique(self, name: str, technique_fn: Callable):
        """
        Register a creativity technique.
        """
        self.creativity_techniques[name] = {
            "function": technique_fn,
            "registered_at": time.time(),
        }

    def define_problem(
        self, description: str, constraints: List[str], goals: List[str], domain: str
    ) -> str:
        """
        Define a problem to be solved.
        """
        problem_id = f"problem_{len(self.problem_history)}"
        problem = {
            "id": problem_id,
            "description": description,
            "constraints": constraints,
            "goals": goals,
            "domain": domain,
            "created_at": time.time(),
            "status": "defined",
        }
        self.problem_history.append(problem)
        return problem_id

    def generate_solutions(
        self, problem_id: str, techniques: List[str] = None, count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple solutions for a problem using creativity techniques.
        """
        # Find the problem
        problem = next((p for p in self.problem_history if p["id"] == problem_id), None)
        if not problem:
            logger.warning(f"Problem {problem_id} not found")
            return []

        # Update problem status
        problem["status"] = "solving"

        # If no specific techniques are requested, use all registered techniques
        if not techniques:
            techniques = list(self.creativity_techniques.keys())

        solutions = []
        for technique_name in techniques:
            if technique_name not in self.creativity_techniques:
                logger.warning(f"Technique {technique_name} not registered")
                continue

            technique = self.creativity_techniques[technique_name]["function"]

            # Apply the technique to generate solutions
            try:
                # In a real system, this would call the actual technique function
                # Here we'll simulate solutions
                for i in range(count):
                    solution = {
                        "id": f"sol_{problem_id}_{technique_name}_{i}",
                        "problem_id": problem_id,
                        "technique": technique_name,
                        "description": f"Solution using {technique_name} (#{i+1})",
                        "details": f"This is a simulated solution {i+1} using {technique_name}",
                        "created_at": time.time(),
                        "novelty": random.uniform(0.3, 0.9),
                        "feasibility": random.uniform(0.4, 0.9),
                    }
                    solutions.append(solution)
            except Exception as e:
                logger.error(f"Error applying technique {technique_name}: {e}")

        # Record the solution approaches
        self.solution_approaches[problem_id] = solutions

        # Update problem status
        problem["status"] = "solutions_generated"

        return solutions

    def evaluate_solution(
        self, solution_id: str, criteria: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluate a solution against multiple criteria.
        """
        # Find the solution
        solution = None
        problem_id = None
        for pid, sols in self.solution_approaches.items():
            for sol in sols:
                if sol["id"] == solution_id:
                    solution = sol
                    problem_id = pid
                    break
            if solution:
                break

        if not solution:
            return {"error": "Solution not found"}

        # Find the problem
        problem = next((p for p in self.problem_history if p["id"] == problem_id), None)
        if not problem:
            return {"error": "Associated problem not found"}

        # Evaluate the solution against each criterion
        scores = {}
        for criterion, weight in criteria.items():
            # In a real system, this would use actual evaluation methods
            # Here we'll simulate scores
            if criterion == "novelty":
                score = solution.get("novelty", random.uniform(0.3, 0.9))
            elif criterion == "feasibility":
                score = solution.get("feasibility", random.uniform(0.4, 0.9))
            else:
                score = random.uniform(0.3, 0.9)

            scores[criterion] = score

        # Calculate weighted score
        weighted_score = sum(
            score * weight for criterion, score in scores.items()
        ) / sum(criteria.values())

        evaluation = {
            "solution_id": solution_id,
            "problem_id": problem_id,
            "criteria": criteria,
            "scores": scores,
            "weighted_score": weighted_score,
            "evaluated_at": time.time(),
        }

        self.evaluation_history.append(evaluation)
        return evaluation

    def refine_solution(
        self, solution_id: str, feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Refine a solution based on feedback.
        """
        # Find the solution
        solution = None
        problem_id = None
        for pid, sols in self.solution_approaches.items():
            for i, sol in enumerate(sols):
                if sol["id"] == solution_id:
                    solution = sol
                    solution_index = i
                    problem_id = pid
                    break
            if solution:
                break

        if not solution:
            return {"error": "Solution not found"}

        # Create a refined version
        refined_id = f"{solution_id}_refined"
        refined_solution = solution.copy()
        refined_solution["id"] = refined_id
        refined_solution["parent_id"] = solution_id
        refined_solution["description"] = f"Refined: {solution['description']}"
        refined_solution["feedback_applied"] = feedback
        refined_solution["created_at"] = time.time()

    def refine_solution(
        self, solution_id: str, feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Refine a solution based on feedback.
        """
        # Find the solution
        solution = None
        problem_id = None
        for pid, sols in self.solution_approaches.items():
            for i, sol in enumerate(sols):
                if sol["id"] == solution_id:
                    solution = sol
                    solution_index = i
                    problem_id = pid
                    break
            if solution:
                break

        if not solution:
            return {"error": "Solution not found"}

        # Create a refined version
        refined_id = f"{solution_id}_refined"
        refined_solution = solution.copy()
        refined_solution["id"] = refined_id
        refined_solution["parent_id"] = solution_id
        refined_solution["description"] = f"Refined: {solution['description']}"
        refined_solution["feedback_applied"] = feedback
        refined_solution["created_at"] = time.time()

        # Apply refinements based on feedback
        if "novelty_feedback" in feedback:
            # Simulate improvement in novelty
            refined_solution["novelty"] = min(
                1.0, solution.get("novelty", 0.5) + random.uniform(0.05, 0.2)
            )

        if "feasibility_feedback" in feedback:
            # Simulate improvement in feasibility
            refined_solution["feasibility"] = min(
                1.0, solution.get("feasibility", 0.5) + random.uniform(0.05, 0.2)
            )

        if "details_feedback" in feedback:
            # Add more details
            refined_solution["details"] = (
                f"{solution.get('details', '')}\n\nRefinements: {feedback.get('details_feedback')}"
            )

        # Add the refined solution to the approaches
        self.solution_approaches[problem_id].append(refined_solution)

        return refined_solution

    def combine_solutions(
        self, solution_ids: List[str], combination_method: str
    ) -> Dict[str, Any]:
        """
        Combine multiple solutions to create a new hybrid solution.
        """
        if len(solution_ids) < 2:
            return {"error": "Need at least two solutions to combine"}

        # Find all solutions
        solutions = []
        problem_ids = set()
        for pid, sols in self.solution_approaches.items():
            for sol in sols:
                if sol["id"] in solution_ids:
                    solutions.append(sol)
                    problem_ids.add(pid)

        if len(solutions) != len(solution_ids):
            return {"error": "Some solutions not found"}

        if len(problem_ids) > 1:
            return {"error": "Solutions must be for the same problem"}

        problem_id = list(problem_ids)[0]

        # Create a combined solution
        combined_id = f"combined_{'_'.join(s['id'] for s in solutions)}"

        # Combine novelty and feasibility (average with a bonus)
        avg_novelty = sum(s.get("novelty", 0.5) for s in solutions) / len(solutions)
        avg_feasibility = sum(s.get("feasibility", 0.5) for s in solutions) / len(
            solutions
        )

        # Add a synergy bonus for combinations
        novelty_bonus = random.uniform(0.05, 0.15)
        feasibility_bonus = random.uniform(
            -0.1, 0.1
        )  # Might be harder to implement combined solutions

        combined_solution = {
            "id": combined_id,
            "problem_id": problem_id,
            "parent_ids": solution_ids,
            "technique": combination_method,
            "description": f"Combined solution using {combination_method}",
            "details": f"This solution combines elements from {len(solutions)} parent solutions",
            "created_at": time.time(),
            "novelty": min(1.0, avg_novelty + novelty_bonus),
            "feasibility": min(1.0, max(0.1, avg_feasibility + feasibility_bonus)),
        }

        # Add details from parent solutions
        for i, sol in enumerate(solutions):
            combined_solution[
                "details"
            ] += f"\n\nFrom solution {i+1}: {sol.get('details', 'No details')}"

        # Add the combined solution to the approaches
        self.solution_approaches[problem_id].append(combined_solution)

        return combined_solution

    def apply_lateral_thinking(
        self, problem_id: str, perspective_shift: str
    ) -> List[Dict[str, Any]]:
        """
        Apply lateral thinking to generate solutions from a different perspective.
        """
        # Find the problem
        problem = next((p for p in self.problem_history if p["id"] == problem_id), None)
        if not problem:
            return {"error": "Problem not found"}

        # Generate solutions using the new perspective
        lateral_solutions = []

        # In a real system, this would apply actual lateral thinking techniques
        # Here we'll simulate the results
        for i in range(3):  # Generate 3 solutions
            solution = {
                "id": f"lateral_{problem_id}_{i}",
                "problem_id": problem_id,
                "technique": "lateral_thinking",
                "perspective": perspective_shift,
                "description": f"Solution from perspective: {perspective_shift} (#{i+1})",
                "details": f"This solution approaches the problem from the perspective of {perspective_shift}",
                "created_at": time.time(),
                "novelty": random.uniform(
                    0.6, 0.95
                ),  # Lateral thinking tends to produce more novel solutions
                "feasibility": random.uniform(0.3, 0.8),  # But might be less feasible
            }
            lateral_solutions.append(solution)

        # Add the solutions to the approaches
        if problem_id in self.solution_approaches:
            self.solution_approaches[problem_id].extend(lateral_solutions)
        else:
            self.solution_approaches[problem_id] = lateral_solutions

        return lateral_solutions

    def get_problem_solving_report(self, problem_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report on the problem-solving process.
        """
        # Find the problem
        problem = next((p for p in self.problem_history if p["id"] == problem_id), None)
        if not problem:
            return {"error": "Problem not found"}

        # Get all solutions for this problem
        solutions = self.solution_approaches.get(problem_id, [])

        # Get evaluations for these solutions
        solution_ids = [s["id"] for s in solutions]
        evaluations = [
            e for e in self.evaluation_history if e["solution_id"] in solution_ids
        ]

        # Find the best solution based on evaluations
        best_solution = None
        best_score = -1

        for solution in solutions:
            # Find evaluations for this solution
            sol_evaluations = [
                e for e in evaluations if e["solution_id"] == solution["id"]
            ]
            if sol_evaluations:
                # Use the most recent evaluation
                latest_eval = max(sol_evaluations, key=lambda e: e["evaluated_at"])
                if latest_eval["weighted_score"] > best_score:
                    best_score = latest_eval["weighted_score"]
                    best_solution = {"solution": solution, "evaluation": latest_eval}

        # Generate statistics
        techniques_used = set(s["technique"] for s in solutions if "technique" in s)
        avg_novelty = (
            sum(s.get("novelty", 0) for s in solutions) / len(solutions)
            if solutions
            else 0
        )
        avg_feasibility = (
            sum(s.get("feasibility", 0) for s in solutions) / len(solutions)
            if solutions
            else 0
        )

        # Create the report
        report = {
            "problem": problem,
            "solutions_count": len(solutions),
            "evaluations_count": len(evaluations),
            "techniques_used": list(techniques_used),
            "average_novelty": avg_novelty,
            "average_feasibility": avg_feasibility,
            "best_solution": best_solution,
            "generated_at": time.time(),
        }

        return report
