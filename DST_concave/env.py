import mo_gymnasium
import numpy as np
 
 
def get_true_reference_pf():
    """
    Return the environment's known Pareto front as (time_return, treasure).

    time_return is negative time cost, so both objectives are maximized:
    higher time_return means lower time cost, and higher treasure is better.
    """
    env = mo_gymnasium.make("deep-sea-treasure-concave-v0")
    pf  = np.array(env.unwrapped.pareto_front(gamma=1.0))
    env.close()
    # env gives (treasure, time_return); time_return is already negative
    pf_max = [(float(pt[1]), float(pt[0])) for pt in pf]
    pf_max.sort(key=lambda p: p[0])
    return pf_max
