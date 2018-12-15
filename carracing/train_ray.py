import numpy as np
import json
from model import make_model
from es import CMAES, SimpleGA, OpenES, PEPG
import argparse
import time
import ray  # @UnresolvedImport
import random

def initialize_settings(sigma_init=0.1, sigma_decay=0.9999):
  global population, filebase, game, model, num_params, es
  population = num_worker * num_worker_trial
  filebase = 'log/'+gamename+'.'+optimizer+'.'+str(num_episode)+'.'+str(population)
  model = make_model()
  num_params = model.param_count
  print("size of model", num_params)

  if optimizer == 'ses':
    ses = PEPG(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_alpha=0.2,
      sigma_limit=0.02,
      elite_ratio=0.1,
      weight_decay=0.005,
      popsize=population)
    es = ses
  elif optimizer == 'ga':
    ga = SimpleGA(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_limit=0.02,
      elite_ratio=0.1,
      weight_decay=0.005,
      popsize=population)
    es = ga
  elif optimizer == 'cma':
    cma = CMAES(num_params,
      sigma_init=sigma_init,
      popsize=population)
    es = cma
  elif optimizer == 'pepg':
    pepg = PEPG(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_alpha=0.20,
      sigma_limit=0.02,
      learning_rate=0.01,
      learning_rate_decay=1.0,
      learning_rate_limit=0.01,
      weight_decay=0.005,
      popsize=population)
    es = pepg
  else:
    oes = OpenES(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_limit=0.02,
      learning_rate=0.01,
      learning_rate_decay=1.0,
      learning_rate_limit=0.01,
      antithetic=antithetic,
      weight_decay=0.005,
      popsize=population)
    es = oes


# def sprint(*args):
#   print(args) # if python3, can do print(*args)
#   sys.stdout.flush()

class Seeder:
  def __init__(self, init_seed=0):
    np.random.seed(init_seed)
    self.limit = np.int32(2**31-1)
  def next_seed(self):
    result = np.random.randint(self.limit)
    return result
  def next_batch(self, batch_size):
    result = np.random.randint(self.limit, size=batch_size).tolist()
    return result

@ray.remote
class Actor(object):
  
  def __init__(self, i):
    print('Creating actor %d' % i)
    self.index = i
    self.model = make_model()
    self.model.make_env()

  def work(self, weights, seed, train_mode=True, max_len=-1):
    print('Running worker %d: seed=%d, num_episodes=%d, max_len=%d' % (self.index, seed, num_episode, max_len))
    self.model.set_model_params(weights)

    reward_list, t_list = self.simulate(train_mode=train_mode, render_mode=False, num_episode=num_episode, seed=seed, max_len=max_len)
    if batch_mode == 'min':
      reward = np.min(reward_list)
    else:
      reward = np.mean(reward_list)
    t = np.mean(t_list)
    return reward, t

  def simulate(self, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):
  
    reward_list = []
    t_list = []
    max_episode_length = 1000
    recording_mode = False
    penalize_turning = False
   
    if train_mode and max_len > 0:
      max_episode_length = max_len
   
    if (seed >= 0):
      random.seed(seed)
      np.random.seed(seed)
      self.model.env.seed(seed)
   
    for i in range(num_episode):
      print('Starting episode %d of %d, max_episode_length=%d' % (i+1, num_episode, max_episode_length))
   
      self.model.reset()
   
      obs = self.model.env.reset()
   
      total_reward = 0.0
   
      random_generated_int = np.random.randint(2**31-1)
   
      filename = "record/"+str(random_generated_int)+".npz"
      recording_mu = []
      recording_logvar = []
      recording_action = []
      recording_reward = [0]
   
      for t in range(max_episode_length):
   
        if render_mode:
          self.model.env.render("human")
        else:
          self.model.env.render('rgb_array')
   
        z, mu, logvar = self.model.encode_obs(obs)
        action = self.model.get_action(z)
   
        recording_mu.append(mu)
        recording_logvar.append(logvar)
        recording_action.append(action)
   
        obs, reward, done, _ = self.model.env.step(action)
   
        extra_reward = 0.0 # penalize for turning too frequently
        if train_mode and penalize_turning:
          extra_reward -= np.abs(action[0])/10.0
          reward += extra_reward
   
        recording_reward.append(reward)
   
        total_reward += reward
   
        if done:
          break
   
      #for recording:
      z, mu, logvar = self.model.encode_obs(obs)
      action = self.model.get_action(z)
      recording_mu.append(mu)
      recording_logvar.append(logvar)
      recording_action.append(action)
    
      recording_mu = np.array(recording_mu, dtype=np.float16)
      recording_logvar = np.array(recording_logvar, dtype=np.float16)
      recording_action = np.array(recording_action, dtype=np.float16)
      recording_reward = np.array(recording_reward, dtype=np.float16)
    
      if not render_mode:
        if recording_mode:
          np.savez_compressed(filename, mu=recording_mu, logvar=recording_logvar, action=recording_action, reward=recording_reward)
    
      if render_mode:
        print("total reward", total_reward, "timesteps", t)
      reward_list.append(total_reward)
      t_list.append(t)
   
    return reward_list, t_list

def evaluate_batch(actors, model_params, max_len=-1):
  # duplicate model_params
  solutions = []
  for _ in range(es.popsize):
    solutions.append(np.copy(model_params))

  seeds = np.arange(es.popsize)

  tasks = []
  for i in range(len(seeds)):
    tasks.append(actors[i].work.remote(solutions[i], seeds[i], train_mode=False, max_len=max_len))
  
  result = ray.get(tasks)
  reward_list_total = np.array(result)
  reward_list = reward_list_total[:, 0] # get rewards
  return np.mean(reward_list)

def master():

  start_time = int(time.time())
  print("training %s, population %d, num_worker %d, num_worker_trial %d" % (gamename, es.popsize, num_worker, num_worker_trial))

  seeder = Seeder(seed_start)

  filename = filebase+'.json'
  filename_log = filebase+'.log.json'
  filename_hist = filebase+'.hist.json'
  filename_hist_best = filebase+'.hist_best.json'
  filename_best = filebase+'.best.json'

  t = 0

  history = []
  history_best = [] # stores evaluation averages every 25 steps or so
  eval_log = []
  best_reward_eval = 0
  best_model_params_eval = None

  max_len = -1 # max time steps (-1 means ignore)
  
  # Initialize ray
  ray.init()
  
  print('Creating actors')
  nrof_workers = 8
  actors = [Actor.remote(i) for i in range(nrof_workers)]

  while True:
    t += 1

    solutions = es.ask()

    if antithetic:
      seeds = seeder.next_batch(int(es.popsize/2))
      seeds = seeds+seeds
    else:
      seeds = seeder.next_batch(es.popsize)

    tasks = []
    for i in range(len(seeds)):
      tasks.append(actors[i].work.remote(solutions[i], seeds[i], train_mode=True, max_len=max_len))
    
    result = ray.get(tasks)
    reward_list_total = np.array(result)

    reward_list = reward_list_total[:, 0] # get rewards

    mean_time_step = int(np.mean(reward_list_total[:, 1])*100)/100. # get average time step
    max_time_step = int(np.max(reward_list_total[:, 1])*100)/100. # get average time step
    avg_reward = int(np.mean(reward_list)*100)/100. # get average time step
    std_reward = int(np.std(reward_list)*100)/100. # get average time step

    es.tell(reward_list)

    es_solution = es.result()
    model_params = es_solution[0] # best historical solution
    _ = es_solution[1] # best reward
    _ = es_solution[2] # best of the current batch
    model.set_model_params(np.array(model_params).round(4))

    r_max = int(np.max(reward_list)*100)/100.
    r_min = int(np.min(reward_list)*100)/100.

    curr_time = int(time.time()) - start_time

    h = (t, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev()*100000)/100000., mean_time_step+1., int(max_time_step)+1)

    if cap_time_mode:
      max_len = 2*int(mean_time_step+1.0)
#     else:
#       max_len = -1

    history.append(h)

    with open(filename, 'wt') as out:
      _ = json.dump([np.array(es.current_param()).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

    with open(filename_hist, 'wt') as out:
      _ = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))

    print('game name=%s  iter=%d  time=%.3f  avg_reward=%.3f  r_min=%.3f  r_max=%.3f  r_std=%.3f  rms_std=%.3f  mean_time_step=%d  max_time_step=%d' % ((gamename,) + h))

    if (t == 1):
      best_reward_eval = avg_reward
    if (t % eval_steps == 0): # evaluate on actual task at hand

      prev_best_reward_eval = best_reward_eval
      model_params_quantized = np.array(es.current_param()).round(4)
      reward_eval = evaluate_batch(actors, model_params_quantized, max_len=-1)
      model_params_quantized = model_params_quantized.tolist()
      improvement = reward_eval - best_reward_eval
      eval_log.append([t, reward_eval, model_params_quantized])
      with open(filename_log, 'wt') as out:
        _ = json.dump(eval_log, out)
      if (len(eval_log) == 1 or reward_eval > best_reward_eval):
        best_reward_eval = reward_eval
        best_model_params_eval = model_params_quantized
      else:
        if retrain_mode:
          print("reset to previous best params, where best_reward_eval = %.3f" % best_reward_eval)
          es.set_mu(best_model_params_eval)
      with open(filename_best, 'wt') as out:
        _ = json.dump([best_model_params_eval, best_reward_eval], out, sort_keys=True, indent=0, separators=(',', ': '))
      # dump history of best
      curr_time = int(time.time()) - start_time
      best_record = [t, curr_time, "improvement", improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval]
      history_best.append(best_record)
      with open(filename_hist_best, 'wt') as out:
        _ = json.dump(history_best, out, sort_keys=False, indent=0, separators=(',', ':'))

      print("Evaluation: iter=%d  time=%.3f  improvement=%.3f  current reward=%.3f  prev best reward=%.3f  best reward=%.3f" % (t, curr_time, improvement, reward_eval, prev_best_reward_eval, best_reward_eval))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                'using pepg, ses, openes, ga, cma'))
  
  parser.add_argument('-o', '--optimizer', type=str, help='ses, pepg, openes, ga, cma.', default='cma')
  parser.add_argument('--num_episode', type=int, default=16, help='num episodes per trial')
  parser.add_argument('--eval_steps', type=int, default=25, help='evaluate every eval_steps step')
  parser.add_argument('-n', '--num_worker', type=int, default=64)
  parser.add_argument('-t', '--num_worker_trial', type=int, help='trials per worker', default=1)
  parser.add_argument('--antithetic', type=int, default=1, help='set to 0 to disable antithetic sampling')
  parser.add_argument('--cap_time', type=int, default=0, help='set to 0 to disable capping timesteps to 2x of average.')
  parser.add_argument('--retrain', type=int, default=0, help='set to 0 to disable retraining every eval_steps if results suck.\n only works w/ ses, openes, pepg.')
  parser.add_argument('-s', '--seed_start', type=int, default=0, help='initial seed')
  parser.add_argument('--sigma_init', type=float, default=0.1, help='sigma_init')
  parser.add_argument('--sigma_decay', type=float, default=0.999, help='sigma_decay')

  args = parser.parse_args()

  global optimizer, num_episode, eval_steps, num_worker, num_worker_trial, antithetic, seed_start, retrain_mode, cap_time_mode, batch_mode, gamename

  optimizer = args.optimizer
  num_episode = args.num_episode
  eval_steps = args.eval_steps
  num_worker = args.num_worker
  num_worker_trial = args.num_worker_trial
  antithetic = (args.antithetic == 1)
  retrain_mode = (args.retrain == 1)
  cap_time_mode= (args.cap_time == 1)
  seed_start = args.seed_start
  batch_mode = 'mean'
  gamename = 'carracing'
  
  initialize_settings(args.sigma_init, args.sigma_decay)
  
  master()
