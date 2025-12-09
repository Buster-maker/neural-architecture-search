# !/usr/bin/python
# -*- coding: utf-8 -*-

from utils import Utils, Log, GPUTools
from population import initialize_population
from evaluate import decode, fitnessEvaluate
from evolve import aconpso
import copy, os, time
import configparser
import numpy as np


def fitness_evaluatebegin(population, curr_gen):
    filenames = []
    population_downscale = []

    for i, particle in enumerate(population):
        print(particle)
        # particle_downscale = [int(dimen)+ round((dimen-int(dimen)+0.01)/2-0.01, 2) if round(dimen-int(dimen),2)>=0.02 else round(dimen//1+0.00,2) for dimen in particle]
        particle = copy.deepcopy(particle)
        filename = decode(particle, curr_gen, i)
        filenames.append(filename)
        population_downscale.append(particle)

    err_set, num_parameters, flops = fitnessEvaluate(filenames, curr_gen, is_test=False, population=population_downscale)
    return err_set, num_parameters, flops

def fitness_evaluate(population, curr_gen):
    filenames = []
    population_downscale = []

    for i, particle in enumerate(population):
        print(particle)
        # particle_downscale = [int(dimen)+ round((dimen-int(dimen)+0.01)/2-0.01, 2) if round(dimen-int(dimen),2)>=0.02 else round(dimen//1+0.00,2) for dimen in particle]
        # particle = copy.deepcopy(particle)

        filename = decode(particle, curr_gen, i)
        filenames.append(filename)
        population_downscale.append(particle)

    err_set, num_parameters, flops = fitnessEvaluate(filenames, curr_gen, is_test=False, population=population_downscale)
    return err_set, num_parameters, flops

def evolve(population, gbest_individual, pbest_individuals, velocity_set_int,velocity_set_float, params):
    offspring = []
    new_velocity_set_int = []
    new_velocity_set_float = []
    for i,particle in enumerate(population):
        new_particle, new_velocity_int,new_velocity_float = aconpso(particle, gbest_individual, pbest_individuals[i], velocity_set_int[i],velocity_set_float[i], params)
        offspring.append(new_particle)
        new_velocity_set_int.append(new_velocity_int)
        new_velocity_set_float.append(new_velocity_float)
    return offspring, new_velocity_set_int,new_velocity_set_float

def update_best_particle(curr_gen,population, err_set, num_parameters, flops, gbest, pbest):
    fitnessSet = [(1 - err_set[i]) * (pow(num_parameters[i] / Tp, wp[int(bool(num_parameters[i] > Tp))]) * pow(flops[i] / Tf, wf[
            int(bool(flops[i] > Tf))])) for i in range(len(population))]
    if not pbest:
        pbest_individuals = copy.deepcopy(population)
        pbest_errSet = copy.deepcopy(err_set)
        pbest_params = copy.deepcopy(num_parameters)
        pbest_flops = copy.deepcopy(flops)
        gbest_individual, gbest_err, gbest_params, gbest_flops, gbest_fitness = getGbest([pbest_individuals, pbest_errSet, pbest_params, pbest_flops])
    else:
        gbest_individual, gbest_err, gbest_params, gbest_flops = gbest
        pbest_individuals, pbest_errSet, pbest_params, pbest_flops = pbest
        pbest_fitnessSet = [(1 - pbest_errSet[i]) * (pow(pbest_params[i] / Tp, wp[int(bool(pbest_params[i] > Tp))]) * pow(
                pbest_flops[i] / Tf, wf[int(bool(pbest_flops[i] > Tf))])) for i in range(len(pbest_individuals))]
        gbest_fitness = (1 - gbest_err) * (pow(gbest_params / Tp, wp[int(bool(gbest_params > Tp))]) * pow(
            gbest_flops / Tf, wf[int(bool(gbest_flops > Tf))]))
        for i,fitness in enumerate(fitnessSet):
            if fitness > pbest_fitnessSet[i]:
                pbest_individuals[i] = copy.deepcopy(population[i])
                pbest_errSet[i] = copy.deepcopy(err_set[i])
                pbest_params[i] = copy.deepcopy(num_parameters[i])
                pbest_flops[i] = copy.deepcopy(flops[i])
            if fitness > gbest_fitness:
                gbest_fitness = copy.deepcopy(fitness)
                gbest_individual = copy.deepcopy(population[i])
                gbest_err = copy.deepcopy(err_set[i])
                gbest_params = copy.deepcopy(num_parameters[i])
                gbest_flops = copy.deepcopy(flops[i])
        # pbest_fitnessSet = [
        #     (1 - pbest_errSet[i]) * (1+pow(pbest_params[i] / Tp, wp[int(bool(pbest_params[i] > Tp))]) * pow(
        #         pbest_flops[i] / Tf, wf[int(bool(pbest_flops[i] > Tf))])) for i in range(len(pbest_individuals))]
        if curr_gen==0:
        
            top_2_indices=np.argsort(pbest_fitnessSet)[-40:][::-1]
            population=np.asarray(population)
            pbest_individuals=np.asarray(pbest_individuals)
            pbest_errSet=np.asarray(pbest_errSet)
            pbest_params=np.asarray(pbest_params)
            pbest_flops=np.asarray(pbest_flops)
        
            pbest_individuals=pbest_individuals[top_2_indices]
            pbest_errSet = pbest_errSet[top_2_indices]
            pbest_params = pbest_params[top_2_indices]
            pbest_flops = pbest_flops[top_2_indices]
            population=population[top_2_indices]
        
            pbest_individuals = list(pbest_individuals)
            pbest_errSet = list(pbest_errSet)
            pbest_params = list(pbest_params)
            pbest_flops = list(pbest_flops)
            population=list(population)
        if curr_gen == 1:
            top_2_indices = np.argsort(pbest_fitnessSet)[-30:][::-1]
            population=np.asarray(population)
            pbest_individuals = np.asarray(pbest_individuals)
            pbest_errSet = np.asarray(pbest_errSet)
            pbest_params = np.asarray(pbest_params)
            pbest_flops = np.asarray(pbest_flops)
        
            pbest_individuals = pbest_individuals[top_2_indices]
            pbest_errSet = pbest_errSet[top_2_indices]
            pbest_params = pbest_params[top_2_indices]
            pbest_flops = pbest_flops[top_2_indices]
            population=population[top_2_indices]
        
            pbest_individuals = list(pbest_individuals)
            pbest_errSet = list(pbest_errSet)
            pbest_params = list(pbest_params)
            pbest_flops = list(pbest_flops)
            population=list(population)
        if curr_gen == 2:
            top_2_indices = np.argsort(pbest_fitnessSet)[-20:][::-1]
            population=np.asarray(population)
            pbest_individuals = np.asarray(pbest_individuals)
            pbest_errSet = np.asarray(pbest_errSet)
            pbest_params = np.asarray(pbest_params)
            pbest_flops = np.asarray(pbest_flops)
        
            pbest_individuals = pbest_individuals[top_2_indices]
            pbest_errSet = pbest_errSet[top_2_indices]
            pbest_params = pbest_params[top_2_indices]
            pbest_flops = pbest_flops[top_2_indices]
            population=population[top_2_indices]
        
            pbest_individuals = list(pbest_individuals)
            pbest_errSet = list(pbest_errSet)
            pbest_params = list(pbest_params)
            pbest_flops = list(pbest_flops)
            population=list(population)
        if curr_gen == 3:
            top_2_indices = np.argsort(pbest_fitnessSet)[-20:][::-1]
            population = np.asarray(population)
            pbest_individuals = np.asarray(pbest_individuals)
            pbest_errSet = np.asarray(pbest_errSet)
            pbest_params = np.asarray(pbest_params)
            pbest_flops = np.asarray(pbest_flops)
        
            pbest_individuals = pbest_individuals[top_2_indices]
            pbest_errSet = pbest_errSet[top_2_indices]
            pbest_params = pbest_params[top_2_indices]
            pbest_flops = pbest_flops[top_2_indices]
            population = population[top_2_indices]
        
            pbest_individuals = list(pbest_individuals)
            pbest_errSet = list(pbest_errSet)
            pbest_params = list(pbest_params)
            pbest_flops = list(pbest_flops)
            population = list(population)
        if curr_gen == 4:
            top_2_indices = np.argsort(pbest_fitnessSet)[-20:][::-1]
            population = np.asarray(population)
            pbest_individuals = np.asarray(pbest_individuals)
            pbest_errSet = np.asarray(pbest_errSet)
            pbest_params = np.asarray(pbest_params)
            pbest_flops = np.asarray(pbest_flops)
        
            pbest_individuals = pbest_individuals[top_2_indices]
            pbest_errSet = pbest_errSet[top_2_indices]
            pbest_params = pbest_params[top_2_indices]
            pbest_flops = pbest_flops[top_2_indices]
            population = population[top_2_indices]
        
            pbest_individuals = list(pbest_individuals)
            pbest_errSet = list(pbest_errSet)
            pbest_params = list(pbest_params)
            pbest_flops = list(pbest_flops)
            population = list(population)
        if curr_gen >= 5:
            top_2_indices = np.argsort(pbest_fitnessSet)[-20:][::-1]
            population=np.asarray(population)
            pbest_individuals = np.asarray(pbest_individuals)
            pbest_errSet = np.asarray(pbest_errSet)
            pbest_params = np.asarray(pbest_params)
            pbest_flops = np.asarray(pbest_flops)
        
            pbest_individuals = pbest_individuals[top_2_indices]
            pbest_errSet = pbest_errSet[top_2_indices]
            pbest_params = pbest_params[top_2_indices]
            pbest_flops = pbest_flops[top_2_indices]
            population=population[top_2_indices]
        
            pbest_individuals = list(pbest_individuals)
            pbest_errSet = list(pbest_errSet)
            pbest_params = list(pbest_params)
            pbest_flops = list(pbest_flops)
            population=list(population)

    return [gbest_individual, gbest_err, gbest_params, gbest_flops], [pbest_individuals, pbest_errSet, pbest_params, pbest_flops],population

def getGbest(pbest):
    pbest_individuals, pbest_errSet, pbest_params, pbest_flops = pbest
    gbest_err = 1.0
    gbest_params = 10e6
    gbest_flops = 10e9
    gbest = None

    gbest_fitness = (1 - gbest_err) * (1+pow(gbest_params / Tp, wp[int(bool(gbest_params > Tp))]) * pow(gbest_flops / Tf,wf[int(bool(gbest_flops > Tf))]))
    pbest_fitnessSet = [(1 - pbest_errSet[i]) *(1+ pow(pbest_params[i] / Tp, wp[int(bool(pbest_params[i] > Tp))])
                        * pow(pbest_flops[i] / Tf, wf[int(bool(pbest_flops[i] > Tf))]))
                        for i in range(len(pbest_individuals))]

    for i,indi in enumerate(pbest_individuals):
        if pbest_fitnessSet[i] > gbest_fitness:
            gbest = copy.deepcopy(indi)
            gbest_err = copy.deepcopy(pbest_errSet[i])
            gbest_params = copy.deepcopy(pbest_params[i])
            gbest_flops = copy.deepcopy(pbest_flops[i])
            gbest_fitness = copy.deepcopy(pbest_fitnessSet[i])
    return gbest, gbest_err, gbest_params, gbest_flops, gbest_fitness

def fitness_test(final_individual):
    final_individual = copy.deepcopy(final_individual)
    new_list = []

    for num in final_individual:
        # 将小数部分提取出来
        decimal_part = num - int(num)

        # 将小数部分变为原来的二倍
        new_decimal_part = decimal_part * 2
        # 如果新的小数部分大于99或者溢出，用三位小数表示
        if  new_decimal_part >= 1:

            new_decimal_part = 0.99

        # 将修改后的小数部分和原来的整数部分组合成新的元素
        new_num = int(num) + new_decimal_part
        # 添加到新的列表中
        new_list.append(new_num)
    filename = Utils.generate_final_pytorch_file(new_list, -1, -1,roundnum=3)
    err_set, num_parameters, flops = fitnessEvaluate([filename], -1, True, [new_list], [32], [weight_decay])
    return err_set[0], num_parameters[0], flops[0]
def evolveCNN(params):
    gen_no = 0
    Log.info('Initialize...')
    start = time.time()
    population = initialize_population(params)

    Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (gen_no))
    err_set, num_parameters, flops  = fitness_evaluate(population, gen_no)
    Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))

    # update gbest and pbest, each individual contains two vectors, vector_archit and vector_conn
    [gbest_individual, gbest_err, gbest_params, gbest_flops], [pbest_individuals, pbest_errSet, pbest_params, pbest_flops],population = update_best_particle(-1,population, err_set, num_parameters, flops, gbest=None, pbest=None)
    Log.info('EVOLVE[%d-gen]-Finish the updating' % (gen_no))

    Utils.save_population_and_err('population', population, err_set, num_parameters, flops, gen_no)
    Utils.save_population_and_err('pbest', pbest_individuals, pbest_errSet, pbest_params, pbest_flops, gen_no)
    Utils.save_population_and_err('gbest', [gbest_individual], [gbest_err], [gbest_params], [gbest_flops], gen_no)

    gen_no += 1
    velocity_set_int = []
    velocity_set_float=[]
    for ii in range(len(population)):
        velocity_set_int.append([0.01] * len(population[ii]))
        velocity_set_float.append([0.01] * len(population[ii]))
    for curr_gen in range(gen_no, params['num_iteration']):
        params['gen_no'] = curr_gen

        Log.info('EVOLVE[%d-gen]-Begin pso evolution' % (curr_gen))
        population, new_velocity_set_int,new_velocity_set_float = evolve(population, gbest_individual, pbest_individuals, velocity_set_int,velocity_set_float, params)
        Log.info('EVOLVE[%d-gen]-Finish pso evolution' % (curr_gen))

        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (curr_gen))
        err_set, num_parameters, flops  = fitness_evaluate(population, curr_gen)
        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (curr_gen))

        [gbest_individual, gbest_err, gbest_params, gbest_flops], [pbest_individuals, pbest_errSet, pbest_params,pbest_flops] ,population= update_best_particle(curr_gen,population, err_set, num_parameters, flops, gbest=[gbest_individual, gbest_err, gbest_params, gbest_flops], pbest=[pbest_individuals, pbest_errSet, pbest_params, pbest_flops])
        Log.info('EVOLVE[%d-gen]-Finish the updating' % (curr_gen))

        Utils.save_population_and_err('population', population, err_set, num_parameters, flops, curr_gen)
        Utils.save_population_and_err('pbest', pbest_individuals, pbest_errSet, pbest_params, pbest_flops, curr_gen)
        Utils.save_population_and_err('gbest', [gbest_individual], [gbest_err], [gbest_params], [gbest_flops], curr_gen)

    end = time.time()
    Log.info('Total Search Time: %.2f seconds' % (end-start))
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    Log.info("%02dh:%02dm:%02ds" % (h, m, s))

    search_time = str("%02dh:%02dm:%02ds" % (h, m, s))
    equipped_gpu_ids, _ = GPUTools._get_equipped_gpu_ids_and_used_gpu_info()
    num_GPUs = len(equipped_gpu_ids)

    proxy_err = copy.deepcopy(gbest_err)

    # final training and test on testset

    gbest_err, num_parameters, flops = fitness_test(gbest_individual)
    Log.info('Error=[%.5f], #parameters=[%d], FLOPs=[%d]' % (gbest_err, gbest_params, gbest_flops))
    Utils.save_population_and_err('final_gbest', [gbest_individual], [gbest_err], [num_parameters], [flops], -1, proxy_err, search_time+', GPUs:%d'%num_GPUs)

def __read_ini_file(section, key):
    config = configparser.ConfigParser()
    config.read('global.ini')
    return config.get(section, key)

if __name__ == '__main__':
    create_directory()
    params = Utils.get_init_params()
    batch_size = int(__read_ini_file('SEARCH', 'batch_size'))
    weight_decay = float(__read_ini_file('SEARCH', 'weight_decay'))
    Tp = float(__read_ini_file('SEARCH', 'Tp'))
    Tf = float(__read_ini_file('SEARCH', 'Tf'))
    wp = list(map(float, __read_ini_file('SEARCH', 'wp').split(',')))
    wf = list(map(float, __read_ini_file('SEARCH', 'wf').split(',')))

    evolveCNN(params)

