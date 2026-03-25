import subprocess


# aspect_ratio = 100



num_rods = 200
total_steps = 2000

# four hours: N200 - 620
# N500 - 80

random_seed = 912
aspect_ratio_list = [25,50,100,200,500]

for aspect_ratio in aspect_ratio_list:
    # subprocess.run(['python', 'run_nudge.py', str(num_rods), str(aspect_ratio), str(random_seed), str(total_steps)])
    subprocess.run(['python', 'run_packing.py', str(aspect_ratio), str(random_seed)])

# python run_nudge.py 200 100 42
