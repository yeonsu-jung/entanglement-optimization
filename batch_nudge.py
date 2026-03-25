import subprocess


# aspect_ratio = 100
random_seed = 42



num_rods = 200
total_steps = 3000

# four hours: N200 - 620
# N500 - 80

aspect_ratio_list = [10,20,50,75,100,150,200,300,500]

for aspect_ratio in aspect_ratio_list:
    subprocess.run(['python', 'run_nudge.py', str(num_rods), str(aspect_ratio), str(random_seed), str(total_steps)])

# python run_nudge.py 200 100 42
