import random

from pandas import DataFrame

from sail_model_optimizer.optimizer.optimizer_base import OptimizerBase


class OptimizerGenetic(OptimizerBase):
    def __init__(self, run_evaluator, run_dict_builder, fold_count=5, optimize_hyper_parameter_count: int = 6) -> None:
        super().__init__(run_evaluator, fold_count, optimize_hyper_parameter_count)
        self.run_dict_builder = run_dict_builder

    # Define the fitness evaluation function
    def evaluate_fitness(self, individual, df_input, df_output, dict_run):
        dict_run["list_feature_selected"] = individual
        return self.evaluate_run_fold(df_input, df_output, dict_run)["score"]

    def genetic_feature_selection(self, df_input, df_output, dict_run, previous_champion):

        # Set the parameters
        population_size = 100  # Number of individuals in each generation
        num_generations = 5  # Number of generations
        mutation_rate = 0.1  # Probability of mutation
        population = []

        if previous_champion != None:
            for _ in range(population_size - 1):
                individual = previous_champion.copy()  # Create a copy of the previous champion

                # Randomly mutate the individual by adding or removing features
                for _ in range(random.randint(1, 3)):  # Adjust the range based on the desired number of mutations
                    if random.random() < 0.5:  # 50% chance of adding a feature
                        feature_to_add = random.choice(df_input.columns)  # Randomly select a feature from X.columns
                        if feature_to_add not in individual:  # Add the feature if it's not already present
                            individual.append(feature_to_add)
                    else:  # 50% chance of removing a feature
                        if len(individual) > 1:  # Ensure at least one feature remains
                            feature_to_remove = random.choice(
                                individual
                            )  # Randomly select a feature from the individual
                            individual.remove(feature_to_remove)  # Remove the feature from the individual
                population.append(list(set(individual)))

        else:  # First run
            # Initialize the population
            for _ in range(population_size):
                individual = random.sample(df_input.columns, random.randint(1, len(df_input.columns)))
                population.append(individual)

        # Main loop
        for generation in range(num_generations):
            print(f"Generation {generation + 1}/{num_generations}")

            # Evaluate fitness for each individual
            fitness_scores = [
                self.evaluate_fitness(individual, df_input, df_output, dict_run) for individual in population
            ]

            # Select parents for reproduction (tournament selection)
            parents = []
            for _ in range(population_size):
                tournament = random.choices(population, k=5)
                parent = max(tournament, key=lambda x: fitness_scores[population.index(x)])
                parents.append(parent)

            # Reproduction (crossover)
            offspring = []
            for i in range(population_size):
                parent1 = parents[i]
                parent2 = parents[(i + 1) % population_size]
                max_crossover_point = min(len(parent1), len(parent2)) - 1
                min_crossover_point = max(1, max_crossover_point // 2)  # Minimum crossover point as half of the maximum
                crossover_point = random.randint(min_crossover_point, max_crossover_point)
                child = parent1[:crossover_point] + parent2[crossover_point:]
                offspring.append(child)

            # Mutation
            for individual in offspring:
                for i in range(len(individual)):
                    if random.random() < mutation_rate:
                        random_index = random.randint(0, len(individual) - 1)
                        individual[i], individual[random_index] = individual[random_index], individual[i]

                if len(individual) == 0:
                    individual.append(random.choice(df_input.columns))
                elif len(individual) > len(df_input.columns):
                    individual = random.sample(individual, random.randint(1, len(df_input.columns)))

            # Update the population with the offspring
            new_offspring = []
            for individual in offspring:
                individual = list(set(individual))
                new_offspring.append(individual)
            offspring = new_offspring

            population = offspring

        # Select the best individual (highest fitness score)
        best_individual = max(population, key=lambda x: self.evaluate_fitness(x, df_input, df_output, dict_run))
        dict_run["list_feature_selected"] = best_individual
        dict_run = self.run_evaluator.evaluate_run(df_input, df_output, df_input, df_output, dict_run)

        return dict_run

    def optimize_model(
        self,
        df_input: DataFrame,
        df_output: DataFrame,
    ) -> dict:
        # initial run
        dict_run = self.run_dict_builder.build_run_dict(df_input, df_output)
        dict_run = self.run_evaluator.evaluate_run(df_input, df_output, df_input, df_output, dict_run)
        score_best = dict_run["score"]
        # iterative run
        has_improvement = True
        hasnt_improved = 0
        # while has_improvement or hasnt_improved < 5:
        while has_improvement:
            has_improvement = False
            hasnt_improved += 1
            # print(f"Starting iteration")
            # print(f"new best score {score_best}")
            print("param count: " + str(len(dict_run["list_feature_selected"])))
            dict_run = self.genetic_feature_selection(df_input, df_output, dict_run, dict_run["list_feature_selected"])
            dict_run = self.optimize_hyper_parameter(df_input, df_output, dict_run)
            if score_best < dict_run["score"]:
                score_best = dict_run["score"]
                print("!!!!param count: " + str(len(dict_run["list_feature_selected"])))
                print(f"!!!!new genetic best score {score_best}")
                has_improvement = True
                hasnt_improved = 0
            print("didn't improve: " + str(hasnt_improved))
        return dict_run