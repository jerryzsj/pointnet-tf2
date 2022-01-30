""" Wrapper functions for evolutionary algorithm

Author: Senjing Zheng
Date: Januray 2021
"""

import copy
import numpy as np
from numpy import random as nr
import os


nr.seed()

def load_npy(file_dir):
    data = np.load(os.path.join(file_dir, 'data.npy'))
    label = np.loadtxt(os.path.join(file_dir, 'label.dat'), dtype=int)
    return data, label

def training_simulation(total_num_genes, data_idx):
    count_miss = np.full((total_num_genes), 0.0)
    count_try = np.full((total_num_genes), 0.0)
    for d in data_idx:
        count_try[d]+=1
        if nr.rand()>0.99:
            count_miss[d]+=1
    return (count_miss, count_try)


##########################################################################################
# Class for GenePool
# fitness is the rate of miss-classification vs total-tries
# fitness is the difficulties to be classified
# higher the fitness, higher the difficulty
##########################################################################################
class GenePool(object):
    """docstring for GenePool"""
    def __init__(self, 
            total_num_genes, # total number of chromo in the pool
            initiated_fitness=1.0, # initiated fitness for each chromo
            memory_len=10, # memory of very last few times of tryings
            history_weighted_scheme = True,
            long_weight=0.8, short_weight=0.2,
            forget_scheme=True, 
            forget_threshold=1 # threshold for forget fitness-sum
        ):
        super(GenePool, self).__init__()
        # print('########################################################################')
        # print('########################################################################')
        # print('Initiate GenePool')
        
        self.total_num_genes = total_num_genes
        # print('self.total_num_genes', self.total_num_genes)
        
        self.initiated_fitness = initiated_fitness
        # print('self.initiated_fitness', self.initiated_fitness)

        # fitness matrix containing fitness of very last few epoch
        self.memory_len = memory_len
        # print('self.memory_len', self.memory_len)
        
        # init fitness_matrix for each chromo, shape@[memory_len, total_num_genes], where the very first one is the latest/newest one
        self.fitness_matrix = np.full((self.memory_len, self.total_num_genes), self.initiated_fitness)
        # print('self.fitness_matrix', self.fitness_matrix.shape)
        # # print(self.fitness_matrix)

        # if to use long-short history weighted fitness
        # if use: fitness = long_w * old__fitness + short_w * new_fitness
        self.history_weighted_scheme = history_weighted_scheme
        # print('self.history_weighted_scheme', self.history_weighted_scheme)
        self.long_weight = long_weight
        # print('self.long_weight', self.long_weight)
        self.short_weight = short_weight
        # print('self.short_weight', self.short_weight)

        # if to forget the very-low-fitness (i.e. totally-correct) patterns/chromos
        self.forget_scheme = forget_scheme
        # print('self.forget_scheme', self.forget_scheme)
        self.forget_threshold = forget_threshold
        # print('self.forget_threshold', self.forget_threshold)

        self.zero_error_string = np.full(self.total_num_genes, 0)
        self.zero_error_list = []

    def get_fitness_matrix(self):
        return(copy.deepcopy(self.fitness_matrix))

    def get_fitness_list(self):
        return(copy.deepcopy(self.fitness_matrix[0]))

    def get_zero_error_list(self):
        return(copy.deepcopy(self.zero_error_list))

    def get_total_num_genes(self):
        return(copy.deepcopy(self.total_num_genes))

    #######################
    # algorithm functions #
    #######################
    # update fitness_matrix with return from network: count_miss & count_try with shape@[N]
    # count_miss & count_try: contain times of mis-classification & tryings in the training, laying with spread of index    
    def update_fitness_matrix(self, count_miss, count_try):
        # print('*************************')
        # print('Update fitness_matrix for GenePool from count_miss & count_try')
        # # print('old fitness matrix')
        # # print(self.fitness_matrix)
        # delete oldest fitness_matrix[-1]
        self.fitness_matrix[1:] = self.fitness_matrix[:-1].copy()
        for i in range(self.total_num_genes):
            # the data was taken into training
            if count_try[i] != 0:
                fts_old = float(self.fitness_matrix[0,i])
                fts_new = count_miss[i] / float(count_try[i])
                if self.history_weighted_scheme:
                    self.fitness_matrix[0,i] = (self.long_weight * fts_old) + (self.short_weight * fts_new)
                else:
                    self.fitness_matrix[0,i] = fts_new
        # print('latest fitness_list for GenePool (fitness for each data-pattern):')
        # print(self.fitness_matrix[0])


    # update perfect chromo
    def update_zero_error_list(self):
        # print('*************************')
        # print('Update zero_error_string & zero_error_list:') 
        # print('if the mean-fitness for a pattern is smaller than the forget_threshold --> forget the pattern')
        self.zero_error_string = np.full(self.total_num_genes, 0)
        self.zero_error_list = []

        sum_fitness = np.mean(self.fitness_matrix, axis=0)
        # # print('sum_fitness')
        # # print(sum_fitness)
        for idx in range(self.total_num_genes):
            # was taken into training
            if sum_fitness[idx] < self.forget_threshold:
                self.zero_error_string[idx] = 1
                self.zero_error_list.append(idx)

        # print('new zero_error_string:')
        # print(self.zero_error_string)
        # print('new zero_error_list:')
        # print('zero_error_list contains idx of pattern which is well-learnt by network')
        # print(self.zero_error_list)
        # # print('new zero_error_string')
        # # print(self.zero_error_string)

    def run_one_epoch(self, count_miss, count_try):
        # print('########################################################################')
        # print('Run one epoch for GenePool after training from input count_miss & count_try')
        # print('will update fitness_matrix & zero_error_list')
        self.update_fitness_matrix(count_miss, count_try)
        if self.forget_scheme:
            self.update_zero_error_list()
# Class for GenePool
##########################################################################################



##########################################################################################
# Class for individuals
###########################
# Tier 1 variable: 'genome_string' to store genome: 
# a binary string: (1) for expressed genes, and 0 for unexpressed genes
###########################
# Tier 2 variable: 'express_genes_list' & 'unexpress_genes_list':
# list of index
###########################
class Individual(object):
    def __init__(self, 
            gene_pool, 
            random_init=True,
            full_init=False,
            genome_string=None, 
            murate_individual=0.3, 
            murate_genes=0.5,
            forget_scheme=True
        ):
        super(Individual, self).__init__()
        # print('************************************************************')
        # print('Initiate Individual')

        self.gene_pool = gene_pool
        # print('inherite GenePool from input GenePool')

        self.genes_fitness_list = gene_pool.get_fitness_list()
        # print('init genes_fitness_list from input GenePool:')
        # # print(self.genes_fitness_list)

        self.total_num_genes = gene_pool.get_total_num_genes()
        # print('init total_num_genes from input GenePool:', self.total_num_genes)

        # initiate genome_string with input-expression or with random-expression/full-expression
        if random_init:
            # print('init genome_string with random scheme')
            self.random_init_genome_string()
            # # print(self.genome_string)
        elif full_init:
            # print('init genome_string with full scheme')
            self.full_init_genome_string()
            # # print(self.genome_string)
        else:
            # print('init genome_string from input genome_string')
            self.genome_string = genome_string
            # # print(self.genome_string)
        # update express_genes_list & unexpress_genes_list
        self.update_express_genes_list()

        # init individual fitness
        self.individual_fitness = np.mean(self.genes_fitness_list[self.express_genes_list])
        # print('updating individual_fitness:', self.individual_fitness)

        # other parms and scheme
        self.murate_individual = murate_individual
        self.murate_genes = murate_genes
        self.forget_scheme = forget_scheme


    ###########################
    # initiate genome_string 
    # the higher the fitness, the higher the chance the pattern will be selected and expressed
    ###########################
    # initiate genome_string with randomly expressed genes
    def random_init_genome_string(self):
        self.genome_string = np.random.randint(2, size=self.total_num_genes)
    ###########################
    # initiate genome_string with fully expressed genes 
    def full_init_genome_string(self):
        self.genome_string = np.array([1,]*(self.total_num_genes))

    ###########################
    # update individual's gene_pool, genes_fitness_list
    def update_gene_pool(self, gene_pool):
        self.gene_pool = gene_pool
        self.genes_fitness_list = self.gene_pool.get_fitness_list()
        # print('updating gene_pool & genes_fitness_list from input gene_pool')
        # # print(self.genes_fitness_list)

    def update_express_genes_list(self):
        self.express_genes_list = []
        self.unexpress_genes_list = []
        # for idx, g in enumerate(self.genome_string):
        for idx in range(self.total_num_genes):
            if self.genome_string[idx]==1:
                self.express_genes_list.append(idx)
            else:
                self.unexpress_genes_list.append(idx)
        # print('updating express_genes_list, have', len(self.express_genes_list), 'expressed genes')
        # print(self.express_genes_list)

    def update_individual_fitness(self):
        # print('updating individual_fitness')
        # print('old individual_fitness')
        # print(self.individual_fitness)
        self.individual_fitness = np.mean(self.genes_fitness_list[self.express_genes_list])
        # print('new individual_fitness')
        # print(self.individual_fitness)

    ###########################
    # Echo functions
    def echo_express_genes_list(self):
        print('express_genes_list',self.express_genes_list)

    ###########################
    # Get functions
    def get_genome_string(self):
        return(copy.deepcopy(self.genome_string))

    def get_express_genes_list(self):
        return(copy.deepcopy(self.express_genes_list))

    def get_fitness(self):
        return(copy.deepcopy(self.individual_fitness))

    ###########################
    # forget zero error genes #
    ###########################
    def run_forget(self):
        # print('**************************')
        # print('Run forget scheme:')
        # get zero_error_list from gene_pool
        zero_error_list = self.gene_pool.get_zero_error_list()
        # print('zero_error_list:')
        # print(zero_error_list)
        # modify genome_string to delete zero_erro idx
        self.genome_string[zero_error_list] = 0
        # update genes list
        self.update_express_genes_list()

    ######################
    # mutation operation #
    ######################
    def run_mutation(self):
        # print('**************************')
        # print('Run mutation scheme:')
        # randomly generate a string of mutation chance
        m_chance = nr.rand(self.gene_pool.total_num_genes)
        
        for idx, m_c in enumerate(m_chance):
            # if mutation chance smaller than mutation rate --> mutate
            if m_c <= self.murate_genes:
                # 0-->1; 1-->0
                self.genome_string[idx] = 1 - self.genome_string[idx]
        # update genes list
        self.update_express_genes_list()
    ##################################################
    # run_one_epoch: where after one training epoch, #
    # to update individual's gene_pool & fitness     #
    ##################################################
    def run_one_epoch(self, gene_pool):
        # print('############################################################')
        # print('Run one epoch for an individual after training: update GenePool and update individual_fitness')
        self.update_gene_pool(gene_pool)
        self.update_individual_fitness()
        # if self.forget_scheme:
        #   self.run_forget()
# Class for individuals
##########################################################################################


##########################################################################################
##############################
# Global functions           #
##############################
# mating two individuals     #
# using two point cross over #
##############################
# including 1: forget_scheme, where:
# forget patterns always been classfied correctly based on 'zero_error_list'
##############################
# including 2: mutation scheme, where:
# depends on murate_individual, to decide if the new-born's genome will mutate
##############################
def run_mating(_main_individual, _minor_individual, gene_pool):
    # print('************************************************************')
    # print('Generating one new-born child')
    _main_string = _main_individual.get_genome_string()
    _minor_string = _minor_individual.get_genome_string()
    # generate two identical random points
    c_1 = nr.randint(gene_pool.total_num_genes)
    c_2 = nr.randint(gene_pool.total_num_genes)
    # c_1 = 1
    # c_2 =10
    while c_1==c_2:
        c_2 = nr.randint(gene_pool.total_num_genes)
    if c_1 > c_2:
        c_1_copy = copy.deepcopy(c_1)
        c_2_copy = copy.deepcopy(c_2)
        c_1 = c_2_copy
        c_2 = c_1_copy

    # swap genome between two points
    for i in range(c_1, c_2+1):
        _main_string[i] = copy.deepcopy(_minor_string[i])

    # generate new child
    child=Individual(gene_pool=gene_pool, genome_string=_main_string, random_init=False, full_init=False, 
        murate_individual=_main_individual.murate_individual, murate_genes=_main_individual.murate_genes)
    
    # run forget: if some genes were perfectly learnt, then delete them from genome_string
    if gene_pool.forget_scheme:
        child.run_forget()
    # run mutation: if generated number smaller than mutation rate
    if nr.rand()<child.murate_individual:
        child.run_mutation()
    # child.echo_express_genes_list()
    return child
##########################################################################################



##########################################################################################
# Class for GA
class GeneticAlgorithm(object):
    """docstring for GeneticAlgorithm."""
    def __init__(self, 
        # ******************* #
        # parms for GenaticAlgorithm
        selection_scheme='fitness_proportionate_selection', #[rank_selection, fitness_proportionate_selection]
        replace_scheme='replace_with_children', #[replace_with_children, replace_with_rank]
        # ******************* #
        # parms for GenePool
        total_num_genes=591, 
        gene_pool_initiated_fitness=1.0, 
        gene_pool_memory_len=10, 
        history_weighted_scheme=True, 
        long_weight=0.8, 
        short_weight=0.2,
        forget_scheme=True, 
        forget_threshold=1, 
        # ******************* #
        # parms for Population
        num_people=10, 
        num_child=9, 
        num_remain=1, 
        # ******************* #
        # parms for Individuals
        individual_init_scheme = 'random',  #[rendom, full]
        murate_individual=0.3, 
        murate_genes=0.3 
        # ******************* #
        ):
        super(GeneticAlgorithm, self).__init__()

        # print('#######################')
        # print('Initiated myGenePool with the following:')
        # print('total_num_genes:', total_num_genes)
        # print('gene_pool_initiated_fitness:', gene_pool_initiated_fitness)
        # print('gene_pool_memory_len:', gene_pool_memory_len)
        # print('history_weighted_scheme:', history_weighted_scheme)
        # print('long_weight:', long_weight)
        # print('short_weight:', short_weight)
        # print('forget_scheme:', forget_scheme)
        # print('forget_threshold:', forget_threshold)
        # print('#######################')
        # print('Initiated myPeople with the following:')
        # print('num_people:', num_people)
        # print('num_child:', num_child)
        # print('selection_scheme:', selection_scheme, ' [rank_selection, fitness_proportionate_selection]')
        # print('replace_scheme:', replace_scheme, ' [replace_with_children, replace_with_rank]')
        # print('#######################')
        # print('Initiated Individuals with the following:')
        # print('individual_init_scheme:', individual_init_scheme, ' [rendom, full]')
        # print('murate_individual:', murate_individual)
        # print('murate_genes:', murate_genes)
        # print('#######################')

        # parms for GenaticAlgorithm
        self.selection_scheme = selection_scheme
        self.replace_scheme = replace_scheme

        # parms for Population
        self.num_people = int(num_people)
        self.num_child = int(num_child)
        self.num_remain = int(num_remain)
        self.num_selection = int(2*num_child)

        # parms for GenePool
        self.total_num_genes = int(total_num_genes)
        self.gene_pool_initiated_fitness = gene_pool_initiated_fitness
        self.gene_pool_memory_len = int(gene_pool_memory_len)

        self.history_weighted_scheme = history_weighted_scheme
        self.long_weight = long_weight
        self.short_weight = short_weight
        self.forget_scheme=forget_scheme
        self.forget_threshold=forget_threshold
        # init GenePool
        self.myGenePool = GenePool(
            total_num_genes=self.total_num_genes, # total number of chromo in the pool
            initiated_fitness=self.gene_pool_initiated_fitness, # initiated fitness for each chromo
            memory_len=self.gene_pool_memory_len, # memory length for gene_pool to restore fitness
            history_weighted_scheme = history_weighted_scheme,
            long_weight=long_weight, short_weight=short_weight,
            forget_scheme=forget_scheme, 
            forget_threshold=forget_threshold # threshold for forget fitness-sum
        )

        # parms for Individuals
        self.individual_init_scheme = individual_init_scheme
        self.murate_individual = murate_individual
        self.murate_genes = murate_genes

        self._init_algorithm()
        # self._init_people()
        # self._init_selection_scheme()

    #######################
    # Echo functions #
    #######################
    def echo_fitness_list(self, add_msg='during training'):
        print('Fitness list ', add_msg)
        print(self.people_fitness_list)


    #######################
    # Initiated functions #
    #######################
    def _init_algorithm(self):
        # print('############################################################')
        # print('############################################################')
        # print('Initiate myGA')
        self._init_people()
        self._init_selection()

        
    # init people with individuals
    def _init_people(self):
        # print('############################################################')
        # print('Initiate myPeople')
        # init myPeople
        self.myPeople =[]
        if self.individual_init_scheme == 'random':
            # print('init myPeople with random scheme')
            # print('individual will init with random genome_string')
            for i in range(self.num_people):
                # print('************************************************************')
                # print('Init myPeople [', i, ']')
                self.myPeople.append(Individual(self.myGenePool, random_init=True, full_init=False, murate_individual=self.murate_individual, murate_genes=self.murate_genes))

        elif self.individual_init_scheme == 'full':
            # print('init myPeople with full scheme')
            # print('individual will init with full genome_string')
            for i in range(self.num_people):
                # print('Init myPeople [', i, ']')
                self.myPeople.append(Individual(self.myGenePool, random_init=False, full_init=True, murate_individual=self.murate_individual, murate_genes=self.murate_genes))

        # update fitness for each individual
        self._update_people_fitness_list()

        
    def _update_people_fitness_list(self):
        # init fitness for each individual
        # print('*************************')
        # print('Update people_fitness_list:')
        self.people_fitness_list = []
        for i in range(len(self.myPeople)):
            self.people_fitness_list.append(self.myPeople[i].get_fitness())
        # print(self.people_fitness_list)


    # init selection && select [num_selection] for mating
    def run_one_epoch_selection(self):
        # print('############################################################')
        # print('Run one epoch of selection, including initiation')
        # init selection scheme
        self._init_selection()
        self._run_selection()

    def run_selection(self):
        # print('run selection')
        self._run_selection()


    # init selection chance with selected scheme
    # 1st: run fitness ranking --> to rank all individuals
    # 2nd: init selection scheme to calculate selection chance for each individual
    # 3rd: init selection wheel for selecting
    def _init_selection(self):
        # print('############################################################')
        # print('Initiate selection scheme')
        # _run_fitness_ranking
        self._run_fitness_ranking()
        # init selection scheme
        if self.selection_scheme=='fitness_proportionate_selection':
            self._init_fitness_proportionate_selection()
        if self.selection_scheme=='rank_selection':
            self._init_rank_selection()
        # init selection wheel
        self._init_selection_wheel()

    def _run_fitness_ranking(self):
        # print('************************************************************')
        # print('Run fitness_ranking')
        # update people fitness list
        self._update_people_fitness_list()
        # sort fitness list
        ranked_idx = np.argsort(self.people_fitness_list)
        ranked_idx = (np.flip(ranked_idx)).tolist()
        # print('ranked_idx:')
        # print(ranked_idx)
        myPeople = []
        for idx in ranked_idx:
            myPeople.append(self.myPeople[idx])
        # print('update myPeople with new ranking individuals') 
        # print('and update people_fitness_list')
        self.myPeople=myPeople
        self._update_people_fitness_list()

    def _init_fitness_proportionate_selection(self):
        # print('*************************')
        # print('Initiate selection_probalibity with fitness_proportionate_selection')
        self.selection_probalibity = []
        sum_fitness = np.sum(self.people_fitness_list)
        for f in self.people_fitness_list:
            self.selection_probalibity.append(f/sum_fitness)
        # print('selection_probalibity:')
        # print(self.selection_probalibity)

    def _init_rank_selection(self):
        # print('*************************')
        # print('Initiate selection_probalibity with rank_selection')
        self.people_rank_list = np.arange(len(self.myPeople), 0, -1)
        # print('people_rank_list:')
        # print(self.people_rank_list)
        sum_rank = np.sum(self.people_rank_list)
        self.selection_probalibity = []
        for r in self.people_rank_list:
            self.selection_probalibity.append(r/sum_rank)
        # print('selection_probalibity:')
        # print(self.selection_probalibity)

    def _init_selection_wheel(self):
        self.selection_wheel = [0, self.selection_probalibity[0]]
        for idx in range(1, len(self.myPeople)):
            selec_prob = copy.deepcopy(self.selection_probalibity[idx])
            self.selection_wheel.append( selec_prob + copy.deepcopy(self.selection_wheel[idx]) )
        # print('*************************')
        # print('Initiate selection_wheel')
        # print(self.selection_wheel)

    def _run_selection(self):
        # print('*************************')
        # print('Run selection')
        self.selected_idx = []
        for i in range(self.num_selection):
            # randomly generate selection chance
            s_prob = nr.rand()
            # compare s_chance with chance_list
            for j in range(len(self.myPeople)):
                # if s_chance larger than select chance on selection_wheel --> pick
                if s_prob >= self.selection_wheel[j] and s_prob <= self.selection_wheel[j+1] :
                    self.selected_idx.append(j)
                    break
            # print(s_prob)
            # # print()
        # print('selected_idx:')
        # print(self.selected_idx)

    def _generate_children(self):
        # generate offsprings by mating two individuals from self.selected_idx
        # the procedure already including Mutation
        # append offsprings into people
        # print('############################################################')
        # print('############################################################')
        # print('Generate children from selected individuals')

        if self.replace_scheme=='replace_with_children':
            # print('and run replace_scheme: replace_with_children')
            remainPeople = []
            for i in range(self.num_remain):
                remainPeople.append(copy.deepcopy(self.myPeople[i]))

            tempPeople = []
            for i in range(self.num_child):
                tempPeople.append(run_mating (self.myPeople[self.selected_idx[2*i]], self.myPeople[self.selected_idx[2*i+1]], self.myGenePool) )
            
            self.myPeople = copy.deepcopy(tempPeople)
            # update fitness for new_individuals
            self._update_people_fitness_list()
            self._run_fitness_ranking()

            # print('delete poor individuals')
            if self.num_child-self.num_people+self.num_remain > 0:
                for i in range(self.num_child-self.num_people+self.num_remain):
                    del self.myPeople[self.num_people-self.num_remain]

            for i in range(self.num_remain):
                self.myPeople.append(copy.deepcopy(remainPeople[i]))

            self._update_people_fitness_list()


        if self.replace_scheme=='replace_with_rank':
            # print('and run replace_scheme: replace_with_rank')
            for i in range(self.num_child):
                self.myPeople.append(run_mating (self.myPeople[self.selected_idx[2*i]], self.myPeople[self.selected_idx[2*i+1]], self.myGenePool) )
            # print('run the fitness_ranking before deleting poor individuals')
            self._update_people_fitness_list()
            self.echo_fitness_list('after generating new children:')
            self._run_fitness_ranking()

            # print('delete poor individuals')
            for i in range(self.num_child):
                del self.myPeople[self.num_people]
            self._update_people_fitness_list()

    ################
    # IN functions #
    ################
    def _update_genepool(self, count_miss, count_try):
        # update_fitness_matrix
        # forget_scheme
        # update_zero_error_list
        self.myGenePool.run_one_epoch(count_miss, count_try)

    def _update_people(self):
        for individual in self.myPeople:
            # update_gene_pool
            # update_individual_fitness
            individual.run_one_epoch(self.myGenePool)


    #################
    # OUT functions #
    #################
    def get_num_people(self):
        return(copy.deepcopy(self.num_people))

    def get_num_selection(self):
        return(copy.deepcopy(self.num_selection))

    def get_data_idx(self):
        # print('len of myPeople',len(self.myPeople))
        self.data_idx = []
        for individual in self.myPeople:
            self.data_idx.extend(individual.get_express_genes_list())
        # print('len of expressed data:', len(self.data_idx))
        return(copy.deepcopy(self.data_idx))

    def get_express_genes_distribution(self):
        # print('get distribution of express_genes')
        self.express_genes_distribution = np.full((self.total_num_genes), 0)
        for idx in self.data_idx:
            self.express_genes_distribution[idx]+=1
        return(self.express_genes_distribution)

    def get_people_fitness_list(self):
        return(copy.deepcopy(self.people_fitness_list))

    #####################
    # Overall functions #
    #####################
    def run_one_epoch(self, count_miss, count_try):
        # update myGenePool and myPeople as well
        self._update_genepool(count_miss, count_try)
        self._update_people()
        # _init_selection and _run_selection to generate selected_idx
        self.run_one_epoch_selection()
        # generate children based on selected_idx, and update myPeople with new people
        self._generate_children()






if __name__ == "__main__":
    #%%
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')
    DATA_DIR = os.path.join(DATA_DIR, 'shapes')
    DATA_DIR = os.path.join(DATA_DIR, 'shapes_luca_clean_norm')

    data, label = load_npy(os.path.join(DATA_DIR, 'train'))

    NUM_POINT = 1000
    BATCH = 100

    total_num_genes=5

    # print('************************************')
    # # print('start time calculating')
    # time_start = time.perf_counter()

    # myGenePool = GenePool(total_num_genes, history_weighted_scheme = True)
    # myGenePool.update_memory_matrix()
    # # print(myGenePool.get_memory_matrix())
    # ft = (myGenePool.get_fitness_matrix())
    #%%
    
    #%%
    myGA = GeneticAlgorithm()
    #%%
    
    #%%
    # myGA.run_one_epoch_selection()
    data_idx = myGA.get_data_idx()
    #%%
    
    #%%
    print(len(data_idx))
    #%%


    # # print(ft)

    # for i in range(20):
    #   count_miss, count_try = training_simulation(total_num_genes, [4,1,5,9,3])
    #   myGenePool.update_fitness_matrix(count_miss, count_try)
    #   ft = (myGenePool.get_fitness_matrix())
    #   # # print(ft[0])
    #   # # print(ft[1])
    #   myGenePool.update_zero_error_list()
    #   # # print(myGenePool.zero_error_list)
    #   # # print(myGenePool.fitness_matrix)
    
    # person=Individual(myGenePool)
    # person1=Individual(myGenePool)
    # person2=Individual(myGenePool)

    # for i in range(20):
    #   count_miss, count_try = training_simulation(total_num_genes, [1,2,3,4]) # 1,2,3,4 were correctely classified
    #   # print('count_miss',count_miss) 
    #   # print('count_try', count_try)
    #   myGenePool.run_one_epoch(count_miss, count_try)

    #   person.run_one_epoch(myGenePool)
    #   person.echo_express_genes_list()
    #   # # print('#######################')
        

    #   # person2=Individual(myGenePool)
    #   # person3 = run_mating(person1, person2, myGenePool)
    #   # person3.echo_express_genes_list()
    #   # print('*****************')


    

    # for i in range(10):   
    #   data_idx = myGA.get_data_idx()
    #   # print('total_num_used_chromo:', len(data_idx))
    #   count_miss, count_try = training_simulation(total_num_genes, data_idx)
    #   myGA.run_one_epoch(count_miss, count_try)
    #   # # print(myGA.get_express_chromo_distribution())


    # time_elapsed = (time.perf_counter() - time_start)
    # # print('using time %.6f' % time_elapsed)
    # # print('************************************')



# # Class for GA
# ##########################################################################################
# def plot_genome_disb(data, num_class=12, num_batch=200):
#   max_row = 4
#   max_col = 3
#   for i in range(1, num_class+1):
#       ax = plt.subplot(max_row, max_col, i)
#       gap_data = []
#       for d in data:
#           gap_data.append(d[(i-1)*200:i*200])

#       plt.imshow(np.array(gap_data), cmap="YlGn")
#       ax.set_title("class id:{}".format(i))
        
#   plt.subplots_adjust(top=0.9, bottom=0.1,left=0.1, right=0.8, hspace=0.4, wspace=0.2)
#   cax = plt.axes([0.85, 0.1, 0.02, 0.8])
#   plt.colorbar(cax=cax)
#   plt.show()

# def pointnet_simulation(label):
#   return nr.rand(label.shape[0])

# def eval_map_simulation(label):
#   # print('size of data:', label.shape[0])
#   eval_map = []
#   for i in range(label.shape[0]):
#       eval_map.append(nr.randint(2))
#   return np.array(eval_map)

# def training_simulation(total_num_genes, data_idx):
#   count_miss = np.full((total_num_genes), 0.0)
#   count_try = np.full((total_num_genes), 0.0)
#   for d in data_idx:
#       count_try[d]+=1
#       if nr.rand()>0.8:
#           count_miss[d]+=1
#   return (count_miss, count_try)
