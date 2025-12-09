# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import copy
import math

def aconpso(particle, gbest, pbest, velocity_set_int,velocity_set_float, params):

    particle_length = params['particle_length']
    max_output_channel = params['max_output_channel']

    integerGbest = []
    # decimal part
    decimalGbest = []
    for i in range(len(gbest)):
        integerGbest.append(int(gbest[i]))
        decimalGbest.append(gbest[i] - int(gbest[i]))
    decimalGbest = [x * 10 for x in decimalGbest]

    integerPbest = []
    # decimal part
    decimalPbest = []
    for i in range(len(pbest)):
        integerPbest.append(int(pbest[i]))
        decimalPbest.append(pbest[i] - int(pbest[i]))
    decimalPbest = [x * 10 for x in decimalPbest]

    integerParticle = []
    decimalParticle = []
    for i in range(len(particle)):
        integerParticle.append(int(particle[i]))
        decimalParticle.append(particle[i] - int(particle[i]))
    decimalParticle = [x * 10 for x in decimalParticle]


    decimalVelocity=[x*10 for x in velocity_set_float]

    cur_len = len(particle)
    # 1.velocity calculation
    w, c1, c2 = 0.7298, 1.49618, 1.49618
    r1 = np.random.random(cur_len)
    r2 = np.random.random(cur_len)

    new_velocity_integer = np.asarray(velocity_set_int) * w + c1 * r1 * (
            np.asarray(integerPbest) - np.asarray(integerParticle)) + c2 * r2 * (
                                   np.asarray(integerGbest) - np.asarray(integerParticle))

    # print(new_velocity_integer,'new_velocity_integer')
    w, c3, c4 = 0.7298, 1.49618, 1.49618
    r3 = np.random.random(cur_len)
    r4 = np.random.random(cur_len)
    new_velocity_decimal = np.asarray(decimalVelocity) * w + c1 * r3 * (
            np.asarray(decimalPbest) - np.asarray(decimalParticle)) + c2 * r4 * (
                                   np.asarray(decimalGbest) - np.asarray(decimalParticle))
    new_velocity_decimal=[y/10 for y in new_velocity_decimal]


    new_particle_integer=np.asarray([x+y for x, y in zip(integerParticle,new_velocity_integer)])
    new_particle_integer=np.clip(new_particle_integer,0,15)
    new_particle_integer=np.round(new_particle_integer,0)

    new_particle_decimal =np.asarray([x/10+y for x,y in zip(decimalParticle,new_velocity_decimal)])
    new_particle_decimal=np.clip(new_particle_decimal,0.00,0.49)

    new_velocity=[round(x,0)+y for x,y in zip(new_velocity_integer,new_velocity_decimal)]
    # 2.particle updating

    new_particle=[x+y for x,y in zip(new_particle_integer,new_particle_decimal)]
    new_particle = [round(par, 2) for par in new_particle ]
    new_velocity = list(new_velocity)
    new_velocity_int=[round(x,0) for x in new_velocity_integer]
    new_velocity_float=[round(y,2) for y in new_velocity_decimal]
    # 3.adjust the value according to some constraints
    subparticle_length = particle_length // 3
    subParticles = [new_particle[0:subparticle_length], new_particle[subparticle_length:2 * subparticle_length],
                    new_particle[2 * subparticle_length:]]

    for j, subParticle in enumerate(subParticles):
        valid_particle = [dimen for dimen in subParticle if 0 <= dimen <= 15.99]
        # condition 1：the number of valid layer (non-strided or strided layer, not identity) must >0
        if len(valid_particle) == 0:
            # if the updated particle has no valid value, let the first dimension value to 0.03 (3*3 DW-sep conv, no.filter=3)
            new_particle[j * subparticle_length] = 0.00

    # 4.outlier handling - maintain the particle and velocity within their valid ranges
    updated_particle1 = []
    for k,par in enumerate(new_particle):
        if (0.00 <= par <= 15.99):
            updated_particle1.append(par)
        elif par > 15.99:
            updated_particle1.append(15.99)
        else:
            updated_particle1.append(0.00)

    updated_particle = []
    for k, par in enumerate(updated_particle1):
        if int(round(par - int(par), 2) * 100) + 1  > max_output_channel:
            updated_particle.append(round(int(par) + float(max_output_channel-1)/100,2))
        else:
            updated_particle.append(par)
    # print(updated_particle,'<---------')
    # print("=======================")
    return updated_particle, new_velocity_int,new_velocity_float

