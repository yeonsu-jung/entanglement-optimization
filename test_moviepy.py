import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoClip

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Create a line object
line, = ax.plot([], [], lw=2)

# Create a function to generate each frame of the video
def make_frame(t):
    # Update the line data
    line.set_data(np.random.rand(100), np.random.rand(100))

    # Clear the plot
    ax.clear()

    # Draw the line object
    line.draw(ax)

    # Return the plot as a numpy array
    return np.array(fig.canvas.renderer.buffer_rgba())

# Create a video clip object
clip = VideoClip(make_frame, duration=5)

# Write the video clip to a file
clip.write_videofile('my_animation.mp4')