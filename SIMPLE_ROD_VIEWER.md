# Simple Rod Viewer

A minimal, standalone OpenGL application that displays two rods without external dependencies beyond OpenGL, GLFW, and GLM.

## Features

- **Clean Implementation**: Self-contained OpenGL visualization with embedded shaders
- **Minimal Dependencies**: Only requires OpenGL, GLFW, and GLM
- **Interactive Camera**: Mouse drag to rotate, scroll to zoom
- **Quality Rendering**: Smooth cylinder geometry with proper lighting and materials
- **Easy to Understand**: Single file with clear structure and comments

## Two Rod Setup

The viewer displays:

- **Blue Rod**: Horizontal rod (2.0 length, 0.1 radius) positioned at (-0.5, 0, 0) along X-axis
- **Orange Rod**: Vertical rod (1.5 length, 0.08 radius) positioned at (0.5, 0, 0) along Y-axis

## Controls

- **Mouse Drag**: Rotate camera around the scene
- **Scroll Wheel**: Zoom in/out (distance: 1.0 to 20.0)
- **ESC**: Exit application

## Building

The project is automatically built with the main CMake configuration:

```bash
./build.sh
./build/simple_rod_viewer
```

## Implementation Details

### Rod Structure

```cpp
struct Rod {
    glm::vec3 center;      // Position in 3D space
    glm::vec3 direction;   // Normalized axis direction
    float length;          // Rod length
    float radius;          // Rod radius
    glm::vec3 color;       // RGB color
};
```

### Camera System

- Orbit camera with yaw/pitch controls
- Automatic position calculation based on spherical coordinates
- Smooth zooming with distance clamping

### Cylinder Mesh Generation

- Procedural cylinder with configurable segments (default 32)
- Proper normals for smooth lighting
- Efficient indexed rendering with GL_TRIANGLES

### Shader Pipeline

- Embedded vertex and fragment shaders (no external files)
- Phong lighting model with ambient, diffuse, and specular components
- Per-vertex normal transformation for correct lighting

### Rod Transformation

The `calculateRodMatrix()` function transforms a unit cylinder to match each rod:

1. Translate to rod center position
2. Rotate to align with rod direction (default Y-axis → rod direction)
3. Scale by rod dimensions (radius, length, radius)

## Code Structure

- **main()**: GLFW setup, render loop, event handling
- **createCylinderMesh()**: Procedural cylinder geometry generation
- **calculateRodMatrix()**: Rod-specific transformation matrix calculation
- **compileShader()** / **createShaderProgram()**: OpenGL shader compilation
- **Event Callbacks**: Mouse and keyboard input handling

## Advantages Over Complex Systems

1. **No External Modules**: Self-contained without physics engines or complex frameworks
2. **Immediate Visual Feedback**: Direct OpenGL rendering without intermediate layers
3. **Easy Customization**: Simple to modify rod positions, colors, or add more rods
4. **Educational Value**: Clear example of OpenGL 3.3+ core profile usage
5. **Fast Compilation**: Minimal dependencies mean quick build times

## Extending the Viewer

To add more rods, simply extend the rods vector in main():

```cpp
std::vector<Rod> rods = {
    Rod(glm::vec3(-0.5f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), 2.0f, 0.1f, glm::vec3(0.3f, 0.7f, 1.0f)),
    Rod(glm::vec3(0.5f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 1.5f, 0.08f, glm::vec3(1.0f, 0.55f, 0.25f)),
    // Add more rods here...
};
```

This viewer serves as an excellent starting point for understanding OpenGL rod visualization without the complexity of full physics engines or external module dependencies.
