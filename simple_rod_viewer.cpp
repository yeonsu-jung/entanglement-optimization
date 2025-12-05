/**
 * @file simple_rod_viewer.cpp
 * @brief Simple Two Rod Visualization without external dependencies
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <cmath>
#include <string>

// Simple Rod structure
struct Rod {
    glm::vec3 center;
    glm::vec3 direction;  // normalized axis direction
    float length;
    float radius;
    glm::vec3 color;
    
    Rod(glm::vec3 pos, glm::vec3 dir, float len, float rad, glm::vec3 col)
        : center(pos), direction(glm::normalize(dir)), length(len), radius(rad), color(col) {}
};

// Camera structure
struct Camera {
    glm::vec3 position{0.0f, 0.0f, 5.0f};
    glm::vec3 target{0.0f, 0.0f, 0.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float yaw = -90.0f;
    float pitch = 0.0f;
    float distance = 5.0f;
    
    glm::mat4 getViewMatrix() const {
        return glm::lookAt(position, target, up);
    }
    
    void updatePosition() {
        float x = target.x + distance * cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        float y = target.y + distance * sin(glm::radians(pitch));
        float z = target.z + distance * sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        position = glm::vec3(x, y, z);
    }
};

// Global variables
Camera camera;
bool firstMouse = true;
double lastX = 400.0, lastY = 300.0;
bool mousePressed = false;

// OpenGL objects
GLuint shaderProgram;
GLuint rodVAO, rodVBO, rodEBO;
std::vector<float> cylinderVertices;
std::vector<unsigned int> cylinderIndices;

// Shader sources
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 rodColor;
uniform vec3 lightPos;
uniform vec3 viewPos;

void main() {
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * rodColor;
    
    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * rodColor;
    
    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0);
    
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}
)";

// Function prototypes
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
unsigned int compileShader(unsigned int type, const char* source);
unsigned int createShaderProgram();
void createCylinderMesh(int segments = 32);
glm::mat4 calculateRodMatrix(const Rod& rod);

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4); // 4x MSAA

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create window
    GLFWwindow* window = glfwCreateWindow(1200, 800, "Simple Two Rod Viewer", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Load OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Enable depth testing and multisampling
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    // Create shader program
    shaderProgram = createShaderProgram();
    if (shaderProgram == 0) {
        std::cerr << "Failed to create shader program" << std::endl;
        return -1;
    }

    // Create cylinder mesh
    createCylinderMesh(32);

    // Create two rods
    std::vector<Rod> rods = {
        Rod(glm::vec3(-0.5f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), 2.0f, 0.1f, glm::vec3(0.3f, 0.7f, 1.0f)), // Blue rod (horizontal)
        Rod(glm::vec3(0.5f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 1.5f, 0.08f, glm::vec3(1.0f, 0.55f, 0.25f))  // Orange rod (vertical)
    };

    std::cout << "Simple Two Rod Viewer" << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  Mouse drag: Rotate camera" << std::endl;
    std::cout << "  Scroll: Zoom in/out" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;
    std::cout << std::endl;

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        // Clear
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use shader program
        glUseProgram(shaderProgram);

        // Set up matrices
        camera.updatePosition();
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1200.0f / 800.0f, 0.1f, 100.0f);

        // Set uniforms that don't change per rod
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, glm::value_ptr(glm::vec3(5.0f, 5.0f, 5.0f)));
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(camera.position));

        // Draw each rod
        glBindVertexArray(rodVAO);
        for (const auto& rod : rods) {
            // Calculate model matrix for this rod
            glm::mat4 model = calculateRodMatrix(rod);
            
            // Set rod-specific uniforms
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
            glUniform3fv(glGetUniformLocation(shaderProgram, "rodColor"), 1, glm::value_ptr(rod.color));
            
            // Draw
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(cylinderIndices.size()), GL_UNSIGNED_INT, 0);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &rodVAO);
    glDeleteBuffers(1, &rodVBO);
    glDeleteBuffers(1, &rodEBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}

unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
        return 0;
    }
    return shader;
}

unsigned int createShaderProgram() {
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    
    if (vertexShader == 0 || fragmentShader == 0) {
        return 0;
    }

    unsigned int program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        return 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}

void createCylinderMesh(int segments) {
    cylinderVertices.clear();
    cylinderIndices.clear();

    // Generate vertices for a unit cylinder (radius=1, height=1, centered at origin)
    // We'll scale and transform it per rod in the model matrix
    
    // Top and bottom centers
    cylinderVertices.insert(cylinderVertices.end(), {0.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f}); // top center (pos + normal)
    cylinderVertices.insert(cylinderVertices.end(), {0.0f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f}); // bottom center
    
    // Side vertices (top ring)
    for (int i = 0; i < segments; ++i) {
        float angle = 2.0f * M_PI * i / segments;
        float x = cos(angle);
        float z = sin(angle);
        cylinderVertices.insert(cylinderVertices.end(), {x, 0.5f, z, x, 0.0f, z}); // position + normal
    }
    
    // Side vertices (bottom ring)
    for (int i = 0; i < segments; ++i) {
        float angle = 2.0f * M_PI * i / segments;
        float x = cos(angle);
        float z = sin(angle);
        cylinderVertices.insert(cylinderVertices.end(), {x, -0.5f, z, x, 0.0f, z}); // position + normal
    }

    // Generate indices
    // Top face
    for (int i = 0; i < segments; ++i) {
        cylinderIndices.insert(cylinderIndices.end(), {
            0, static_cast<unsigned int>(2 + i), static_cast<unsigned int>(2 + (i + 1) % segments)
        });
    }
    
    // Bottom face
    for (int i = 0; i < segments; ++i) {
        cylinderIndices.insert(cylinderIndices.end(), {
            1, static_cast<unsigned int>(2 + segments + (i + 1) % segments), static_cast<unsigned int>(2 + segments + i)
        });
    }
    
    // Side faces
    for (int i = 0; i < segments; ++i) {
        int next = (i + 1) % segments;
        unsigned int topCurrent = 2 + i;
        unsigned int topNext = 2 + next;
        unsigned int bottomCurrent = 2 + segments + i;
        unsigned int bottomNext = 2 + segments + next;
        
        // Two triangles per side face
        cylinderIndices.insert(cylinderIndices.end(), {
            topCurrent, bottomCurrent, topNext,
            topNext, bottomCurrent, bottomNext
        });
    }

    // Create OpenGL objects
    glGenVertexArrays(1, &rodVAO);
    glGenBuffers(1, &rodVBO);
    glGenBuffers(1, &rodEBO);

    glBindVertexArray(rodVAO);

    glBindBuffer(GL_ARRAY_BUFFER, rodVBO);
    glBufferData(GL_ARRAY_BUFFER, cylinderVertices.size() * sizeof(float), cylinderVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rodEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cylinderIndices.size() * sizeof(unsigned int), cylinderIndices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

glm::mat4 calculateRodMatrix(const Rod& rod) {
    // Start with identity matrix
    glm::mat4 model = glm::mat4(1.0f);
    
    // Translate to rod center
    model = glm::translate(model, rod.center);
    
    // Rotate to align with rod direction
    // Default cylinder is along Y-axis, we need to rotate to rod.direction
    glm::vec3 defaultDir = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 targetDir = glm::normalize(rod.direction);
    
    // Calculate rotation
    float dot = glm::dot(defaultDir, targetDir);
    if (abs(dot - 1.0f) < 1e-6f) {
        // Already aligned, no rotation needed
    } else if (abs(dot + 1.0f) < 1e-6f) {
        // Opposite direction, rotate 180° around any perpendicular axis
        glm::vec3 rotAxis = glm::vec3(1.0f, 0.0f, 0.0f);
        model = glm::rotate(model, glm::radians(180.0f), rotAxis);
    } else {
        // General case
        glm::vec3 rotAxis = glm::normalize(glm::cross(defaultDir, targetDir));
        float angle = acos(glm::clamp(dot, -1.0f, 1.0f));
        model = glm::rotate(model, angle, rotAxis);
    }
    
    // Scale by rod dimensions
    model = glm::scale(model, glm::vec3(rod.radius, rod.length, rod.radius));
    
    return model;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!mousePressed) {
        lastX = xpos;
        lastY = ypos;
        return;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    camera.yaw += xoffset;
    camera.pitch += yoffset;

    // Constrain pitch
    if (camera.pitch > 89.0f)
        camera.pitch = 89.0f;
    if (camera.pitch < -89.0f)
        camera.pitch = -89.0f;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            glfwGetCursorPos(window, &lastX, &lastY);
        } else if (action == GLFW_RELEASE) {
            mousePressed = false;
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.distance -= (float)yoffset * 0.2f;
    if (camera.distance < 1.0f)
        camera.distance = 1.0f;
    if (camera.distance > 20.0f)
        camera.distance = 20.0f;
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}