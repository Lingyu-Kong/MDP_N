from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
calc=LennardJones(rc=500)

def relax(xyz,max_steps):
    N=len(xyz)
    atm=Atoms('Ar'+str(N),positions=xyz)
    atm.calc=calc
    dyn = BFGS(atm,logfile=None)
    dyn.run(fmax=0.0001,steps=max_steps)
    return dyn.get_number_of_steps(),min(dyn.atoms.get_potential_energy(),10),dyn.atoms.get_positions()

def compute(xyz):
    N=len(xyz)
    atm=Atoms('Ar'+str(N),positions=xyz)
    atm.calc=calc
    return min(atm.get_potential_energy(),10)

def batch_relax(conforms,max_steps,batch_size):
    batch_steps=[]
    batch_energy=[]
    batch_pos=[]
    for i in range(batch_size):
        pos=conforms[i].tolist()
        steps,energy,pos=relax(pos,max_steps)
        batch_steps.append(steps)
        batch_energy.append(energy)
        batch_pos.append(list(map(list, pos)))
    return batch_steps,batch_energy,batch_pos

def batch_compute(conforms,batch_size):
    batch_energy=[]
    for i in range(batch_size):
        pos=list(map(tuple, conforms[i].tolist()))
        energy=compute(pos)
        batch_energy.append(energy)
    return batch_energy