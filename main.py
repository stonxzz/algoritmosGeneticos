import random
from deap import base, creator, tools

# Definir el problema de minimización con representación binaria
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Crear una función que genere individuos en binario (por ejemplo, longitud 8 bits)
def create_individual():
    return [random.randint(0, 1) for _ in range(8)]  # 8 genes en binario

# Función de aptitud que convierte el binario a un número decimal
# La función de aptitud es: f(x) = (x + 6)^2 - 18
def fitness_function(individual):
    # Convertir el individuo (binario) a decimal
    x = int("".join(map(str, individual)), 2)
    # Calcular la aptitud (menor valor es mejor)
    return ((x + 6) ** 2 - 18,)

# Configurar el toolbox
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxTwoPoint)  # Cruzamiento de dos puntos para binario
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Mutación de bit
toolbox.register("select", tools.selTournament, tournsize=3)

# Configurar parámetros del algoritmo genético
population = toolbox.population(n=8)  # Población inicial de 8 individuos
probab_crossover = 0.7  # Probabilidad de cruzamiento
probab_mutate = 0.2     # Probabilidad de mutación
num_generations = 3    # Número de generaciones

# Función para imprimir la población de una generación
def print_population(population, generation):
    print(f"\nGeneración {generation}:")
    for ind in population:
        binary_str = ''.join(map(str, ind))
        decimal_value = int(binary_str, 2)
        print(f"Individuo: {binary_str}, Decimal: {decimal_value}, Aptitud: {ind.fitness.values[0]}")

# Ejecutar el algoritmo
for gen in range(num_generations):
    # Evaluar la aptitud de cada individuo
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Mostrar la población actual
    print_population(population, gen)

    # Seleccionar el mejor, el peor y dos individuos aleatorios
    best_individual = tools.selBest(population, 1)[0]
    worst_individual = tools.selWorst(population, 1)[0]
    random_individuals = random.sample(population, 2)

    offspring = []

    # Cruzamiento "Mejor con Mejor"
    if random.random() < probab_crossover:
        child1, child2 = toolbox.clone(best_individual), toolbox.clone(best_individual)
        toolbox.mate(child1, child2)
        offspring.extend([child1, child2])

    # Cruzamiento "Mejor con Peor"
    if random.random() < probab_crossover:
        child1, child2 = toolbox.clone(best_individual), toolbox.clone(worst_individual)
        toolbox.mate(child1, child2)
        offspring.extend([child1, child2])

    # Cruzamiento "Aleatorio"
    if random.random() < probab_crossover:
        child1, child2 = toolbox.clone(random_individuals[0]), toolbox.clone(random_individuals[1])
        toolbox.mate(child1, child2)
        offspring.extend([child1, child2])

    # Mutación
    for mutant in offspring:
        if random.random() < probab_mutate:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluar la aptitud de los nuevos individuos
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Reemplazar la población por los nuevos hijos (offspring)
    population[:] = tools.selBest(population + offspring, len(population))

# Mostrar la última generación (la mejor obtenida)
print_population(population, num_generations)

# Obtener el mejor individuo
best_individual = tools.selBest(population, 1)[0]
best_binary_str = ''.join(map(str, best_individual))
best_decimal_value = int(best_binary_str, 2)
print(f"\nMejor individuo de todas las generaciones:")
print(f"Individuo: {best_binary_str}, Decimal: {best_decimal_value}, Aptitud: {best_individual.fitness.values[0]}")
