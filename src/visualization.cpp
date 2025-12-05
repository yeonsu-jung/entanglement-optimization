/**
 * @file visualization.cpp
 * @brief Implementation of OpenGL visualization
 */

#include "visualization.hpp"
#include "distance.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <filesystem>

#define GL_SILENCE_DEPRECATION  // Silence OpenGL deprecation warnings on macOS
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace entanglement {

// Vertex shader source
const char* vertex_shader_source = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 uMVP;
uniform vec3 uLightDir;

out vec3 FragPos;
out vec3 Normal;
out float LightIntensity;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    FragPos = aPos;
    Normal = aNormal;
    LightIntensity = max(dot(normalize(aNormal), normalize(uLightDir)), 0.3);
}
)";

// Fragment shader source
const char* fragment_shader_source = R"(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in float LightIntensity;

uniform vec3 uColor;

out vec4 FragColor;

void main() {
    vec3 color = uColor * LightIntensity;
    FragColor = vec4(color, 1.0);
}
)";

RodVisualizer::RodVisualizer(int width, int height, const std::string& title)
    : window_(nullptr), window_width_(width), window_height_(height),
      mouse_pressed_(false), frame_count_(0), last_time_(0.0) {
    
    // Initialize GLFW
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    
    // Set OpenGL version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    // Create window
    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    
    glfwMakeContextCurrent(window_);
    glfwSetWindowUserPointer(window_, this);
    
    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD");
    }
    
    setup_opengl();
    setup_callbacks();
}

RodVisualizer::~RodVisualizer() {
    if (window_) {
        // Cleanup OpenGL objects
        glDeleteProgram(shader_program_);
        glDeleteVertexArrays(1, &rod_vao_);
        glDeleteBuffers(1, &rod_vbo_);
        glDeleteBuffers(1, &rod_ebo_);
        glDeleteVertexArrays(1, &axes_vao_);
        glDeleteBuffers(1, &axes_vbo_);
        glDeleteVertexArrays(1, &container_vao_);
        glDeleteBuffers(1, &container_vbo_);
        
        glfwTerminate();
    }
}

void RodVisualizer::setup_opengl() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glDepthFunc(GL_LESS);
    
    // Set viewport
    glViewport(0, 0, window_width_, window_height_);
    
    create_shaders();
    create_rod_geometry();
    create_axes_geometry();
    create_container_geometry();
}

void RodVisualizer::create_shaders() {
    // Compile vertex shader
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
    glCompileShader(vertex_shader);
    
    // Check vertex shader compilation
    GLint success;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(vertex_shader, 512, nullptr, info_log);
        throw std::runtime_error("Vertex shader compilation failed: " + std::string(info_log));
    }
    
    // Compile fragment shader
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
    glCompileShader(fragment_shader);
    
    // Check fragment shader compilation
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(fragment_shader, 512, nullptr, info_log);
        throw std::runtime_error("Fragment shader compilation failed: " + std::string(info_log));
    }
    
    // Create shader program
    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vertex_shader);
    glAttachShader(shader_program_, fragment_shader);
    glLinkProgram(shader_program_);
    
    // Check program linking
    glGetProgramiv(shader_program_, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(shader_program_, 512, nullptr, info_log);
        throw std::runtime_error("Shader program linking failed: " + std::string(info_log));
    }
    
    // Clean up shaders
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    
    // Get uniform locations
    u_mvp_matrix_ = glGetUniformLocation(shader_program_, "uMVP");
    u_color_ = glGetUniformLocation(shader_program_, "uColor");
    u_light_dir_ = glGetUniformLocation(shader_program_, "uLightDir");
}

void RodVisualizer::create_rod_geometry() {
    // Create cylinder geometry for rods
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    
    const int segments = view_settings_.rod_segments;
    const float radius = view_settings_.rod_radius;
    const float half_length = 0.5f;  // Rod extends from -0.5 to +0.5 in local coordinates
    
    // Generate cylinder vertices
    for (int i = 0; i < segments; ++i) {  // Changed: < segments instead of <= segments
        float angle = 2.0f * M_PI * i / segments;
        float x = radius * cos(angle);
        float y = radius * sin(angle);
        float nx = cos(angle);
        float ny = sin(angle);
        
        // Bottom circle
        vertices.insert(vertices.end(), {x, y, -half_length, nx, ny, 0.0f});
        // Top circle
        vertices.insert(vertices.end(), {x, y, half_length, nx, ny, 0.0f});
    }
    
    // Generate indices for cylinder sides
    for (int i = 0; i < segments; ++i) {
        int bottom1 = i * 2;
        int top1 = i * 2 + 1;
        int bottom2 = ((i + 1) % segments) * 2;  // Fixed: % segments instead of % (segments + 1)
        int top2 = ((i + 1) % segments) * 2 + 1;
        
        // Two triangles per quad
        indices.insert(indices.end(), {
            static_cast<unsigned int>(bottom1), 
            static_cast<unsigned int>(bottom2), 
            static_cast<unsigned int>(top1)
        });
        indices.insert(indices.end(), {
            static_cast<unsigned int>(top1), 
            static_cast<unsigned int>(bottom2), 
            static_cast<unsigned int>(top2)
        });
    }
    
    // Create VAO, VBO, EBO
    glGenVertexArrays(1, &rod_vao_);
    glGenBuffers(1, &rod_vbo_);
    glGenBuffers(1, &rod_ebo_);
    
    glBindVertexArray(rod_vao_);
    
    glBindBuffer(GL_ARRAY_BUFFER, rod_vbo_);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rod_ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void RodVisualizer::create_axes_geometry() {
    // Create coordinate axes
    float axes_vertices[] = {
        // X axis (red)
        0.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        // Y axis (green)
        0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f,
        // Z axis (blue)
        0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f,
    };
    
    glGenVertexArrays(1, &axes_vao_);
    glGenBuffers(1, &axes_vbo_);
    
    glBindVertexArray(axes_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, axes_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(axes_vertices), axes_vertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void RodVisualizer::create_container_geometry() {
    // Create wireframe box
    float box_vertices[] = {
        // Bottom face
        -1.0f, -1.0f, -1.0f,  0.0f, 0.0f, -1.0f,
         1.0f, -1.0f, -1.0f,  0.0f, 0.0f, -1.0f,
         1.0f,  1.0f, -1.0f,  0.0f, 0.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,  0.0f, 0.0f, -1.0f,
        // Top face
        -1.0f, -1.0f,  1.0f,  0.0f, 0.0f,  1.0f,
         1.0f, -1.0f,  1.0f,  0.0f, 0.0f,  1.0f,
         1.0f,  1.0f,  1.0f,  0.0f, 0.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,  0.0f, 0.0f,  1.0f,
    };
    
    glGenVertexArrays(1, &container_vao_);
    glGenBuffers(1, &container_vbo_);
    
    glBindVertexArray(container_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, container_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(box_vertices), box_vertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void RodVisualizer::setup_callbacks() {
    glfwSetMouseButtonCallback(window_, mouse_button_callback);
    glfwSetCursorPosCallback(window_, cursor_position_callback);
    glfwSetScrollCallback(window_, scroll_callback);
    glfwSetKeyCallback(window_, key_callback);
    glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
    glfwSetWindowCloseCallback(window_, window_close_callback);
}

bool RodVisualizer::should_close() const {
    return glfwWindowShouldClose(window_);
}

void RodVisualizer::render(const std::vector<Rod>& rods, double container_size) {
    glClearColor(view_settings_.background_color[0], 
                 view_settings_.background_color[1], 
                 view_settings_.background_color[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glUseProgram(shader_program_);
    
    // Calculate MVP matrix
    float mvp[16];
    calculate_mvp_matrix(mvp);
    glUniformMatrix4fv(u_mvp_matrix_, 1, GL_FALSE, mvp);
    
    // Set light direction
    float light_dir[3] = {1.0f, 1.0f, 1.0f};
    glUniform3fv(u_light_dir_, 1, light_dir);
    
    // Render components
    if (view_settings_.show_axes) {
        render_axes();
    }
    
    if (view_settings_.show_container && container_size > 0.0) {
        render_container(container_size);
    }
    
    render_rods(rods);
    
    if (view_settings_.show_contacts) {
        render_contacts(rods);
    }
}

void RodVisualizer::update() {
    update_camera();
    glfwSwapBuffers(window_);
    glfwPollEvents();
    frame_count_++;
}

void RodVisualizer::update_camera() {
    if (anim_settings_.auto_rotate) {
        double current_time = glfwGetTime();
        if (last_time_ > 0.0) {
            view_settings_.camera_theta += anim_settings_.rotation_speed * (current_time - last_time_);
        }
        last_time_ = current_time;
    }
}

void RodVisualizer::calculate_mvp_matrix(float* mvp) {
    // Create view matrix (camera)
    float eye_x = view_settings_.camera_distance * sin(view_settings_.camera_phi) * cos(view_settings_.camera_theta);
    float eye_y = view_settings_.camera_distance * sin(view_settings_.camera_phi) * sin(view_settings_.camera_theta);
    float eye_z = view_settings_.camera_distance * cos(view_settings_.camera_phi);
    
    // Simple lookAt matrix calculation
    // For simplicity, we'll create a basic perspective projection and view transform
    // This is a simplified implementation - in production you'd use a proper math library
    
    float aspect = float(window_width_) / float(window_height_);
    float fov = M_PI / 4.0f;  // 45 degrees
    float near_plane = 0.1f;
    float far_plane = 100.0f;
    
    // Create identity matrix and apply transformations
    for (int i = 0; i < 16; ++i) mvp[i] = 0.0f;
    mvp[0] = mvp[5] = mvp[10] = mvp[15] = 1.0f;
    
    // Apply perspective projection (simplified)
    float f = 1.0f / tan(fov / 2.0f);
    mvp[0] = f / aspect;
    mvp[5] = f;
    mvp[10] = -(far_plane + near_plane) / (far_plane - near_plane);
    mvp[11] = -1.0f;
    mvp[14] = -(2.0f * far_plane * near_plane) / (far_plane - near_plane);
    mvp[15] = 0.0f;
    
    // Apply camera transform (simplified - translate back)
    mvp[14] -= view_settings_.camera_distance;
}

void RodVisualizer::calculate_rod_mvp_matrix(const Rod& rod, float* mvp) {
    // First, calculate the base view-projection matrix
    calculate_mvp_matrix(mvp);
    
    // Create model matrix using simple approach
    float cos_phi = cos(rod.phi);
    float sin_phi = sin(rod.phi);
    float cos_theta = cos(rod.theta);
    float sin_theta = sin(rod.theta);
    
    // Rod direction vector in spherical coordinates
    // phi=0 is +Z, phi=π/2 is XY plane
    float dx = sin_phi * cos_theta;
    float dy = sin_phi * sin_theta;
    float dz = cos_phi;
    
    // Create transformation matrix for rod
    // Scale by length along Z axis, then rotate to proper orientation, then translate
    float model[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, static_cast<float>(rod.length), 0.0f,
        static_cast<float>(rod.center.x), static_cast<float>(rod.center.y), static_cast<float>(rod.center.z), 1.0f
    };
    
    // Create rotation matrix to align rod with direction vector
    // We want to rotate from (0,0,1) to (dx,dy,dz)
    float up_x = 0.0f, up_y = 1.0f, up_z = 0.0f;  // Default up vector
    
    // Create right vector: right = up x direction
    float right_x = up_y * dz - up_z * dy;
    float right_y = up_z * dx - up_x * dz;
    float right_z = up_x * dy - up_y * dx;
    
    // Normalize right vector
    float right_len = sqrt(right_x*right_x + right_y*right_y + right_z*right_z);
    if (right_len > 1e-6f) {
        right_x /= right_len;
        right_y /= right_len;
        right_z /= right_len;
    } else {
        // Direction is parallel to up, use a different up vector
        right_x = 1.0f; right_y = 0.0f; right_z = 0.0f;
    }
    
    // Recompute up vector: up = direction x right
    up_x = dy * right_z - dz * right_y;
    up_y = dz * right_x - dx * right_z;
    up_z = dx * right_y - dy * right_x;
    
    // Build rotation matrix (right, up, direction as columns)
    float rotation[16] = {
        right_x, up_x, dx, 0.0f,
        right_y, up_y, dy, 0.0f,
        right_z, up_z, dz, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    // Multiply model = rotation * scale_translate
    float final_model[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            final_model[i*4 + j] = 0.0f;
            for (int k = 0; k < 4; ++k) {
                final_model[i*4 + j] += rotation[i*4 + k] * model[k*4 + j];
            }
        }
    }
    
    // Multiply mvp = mvp * final_model
    float result[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i*4 + j] = 0.0f;
            for (int k = 0; k < 4; ++k) {
                result[i*4 + j] += mvp[i*4 + k] * final_model[k*4 + j];
            }
        }
    }
    
    // Copy result back to mvp
    for (int i = 0; i < 16; ++i) mvp[i] = result[i];
}

void RodVisualizer::render_rods(const std::vector<Rod>& rods) {
    glUniform3fv(u_color_, 1, view_settings_.rod_color);
    
    for (const auto& rod : rods) {
        // Calculate MVP matrix for this specific rod
        float mvp[16];
        calculate_rod_mvp_matrix(rod, mvp);
        glUniformMatrix4fv(u_mvp_matrix_, 1, GL_FALSE, mvp);
        
        glBindVertexArray(rod_vao_);
        glDrawElements(GL_TRIANGLES, view_settings_.rod_segments * 6, GL_UNSIGNED_INT, 0);
    }
    
    glBindVertexArray(0);
}

void RodVisualizer::render_axes() {
    float axes_color[3] = {1.0f, 1.0f, 1.0f};
    glUniform3fv(u_color_, 1, axes_color);
    
    glBindVertexArray(axes_vao_);
    glLineWidth(3.0f);
    glDrawArrays(GL_LINES, 0, 6);
    glBindVertexArray(0);
}

void RodVisualizer::render_container(double size) {
    float container_color[3] = {0.5f, 0.5f, 0.5f};
    glUniform3fv(u_color_, 1, container_color);
    
    glBindVertexArray(container_vao_);
    glLineWidth(1.0f);
    // Draw wireframe box
    unsigned int box_indices[] = {
        0, 1, 1, 2, 2, 3, 3, 0,  // Bottom face
        4, 5, 5, 6, 6, 7, 7, 4,  // Top face
        0, 4, 1, 5, 2, 6, 3, 7   // Vertical edges
    };
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, box_indices);
    glBindVertexArray(0);
}

void RodVisualizer::render_contacts(const std::vector<Rod>& rods) {
    glUniform3fv(u_color_, 1, view_settings_.contact_color);
    
    // Highlight rods that are in contact
    for (size_t i = 0; i < rods.size(); ++i) {
        bool has_contact = false;
        for (size_t j = i + 1; j < rods.size(); ++j) {
            if (rod_distance(rods[i], rods[j]) < view_settings_.contact_threshold) {
                has_contact = true;
                break;
            }
        }
        
        if (has_contact) {
            // Render this rod in contact color
            glBindVertexArray(rod_vao_);
            glDrawElements(GL_TRIANGLES, view_settings_.rod_segments * 6, GL_UNSIGNED_INT, 0);
        }
    }
    
    glBindVertexArray(0);
}

void RodVisualizer::save_frame(const std::string& filename) {
    std::vector<unsigned char> pixels(window_width_ * window_height_ * 3);
    glReadPixels(0, 0, window_width_, window_height_, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    
    // Flip image vertically (OpenGL has origin at bottom-left)
    std::vector<unsigned char> flipped(window_width_ * window_height_ * 3);
    for (int y = 0; y < window_height_; ++y) {
        for (int x = 0; x < window_width_; ++x) {
            int src_idx = (y * window_width_ + x) * 3;
            int dst_idx = ((window_height_ - 1 - y) * window_width_ + x) * 3;
            flipped[dst_idx] = pixels[src_idx];
            flipped[dst_idx + 1] = pixels[src_idx + 1];
            flipped[dst_idx + 2] = pixels[src_idx + 2];
        }
    }
    
    stbi_write_png(filename.c_str(), window_width_, window_height_, 3, flipped.data(), window_width_ * 3);
}

// Callback implementations
void RodVisualizer::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    RodVisualizer* viz = static_cast<RodVisualizer*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        viz->mouse_pressed_ = (action == GLFW_PRESS);
        if (viz->mouse_pressed_) {
            glfwGetCursorPos(window, &viz->last_mouse_x_, &viz->last_mouse_y_);
        }
    }
}

void RodVisualizer::cursor_position_callback(GLFWwindow* window, double x, double y) {
    RodVisualizer* viz = static_cast<RodVisualizer*>(glfwGetWindowUserPointer(window));
    if (viz->mouse_pressed_) {
        float dx = x - viz->last_mouse_x_;
        float dy = y - viz->last_mouse_y_;
        
        viz->view_settings_.camera_theta += dx * 0.01f;
        viz->view_settings_.camera_phi += dy * 0.01f;
        
        // Clamp phi to avoid gimbal lock
        viz->view_settings_.camera_phi = std::max(0.1f, std::min((float)M_PI - 0.1f, viz->view_settings_.camera_phi));
        
        viz->last_mouse_x_ = x;
        viz->last_mouse_y_ = y;
    }
}

void RodVisualizer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    RodVisualizer* viz = static_cast<RodVisualizer*>(glfwGetWindowUserPointer(window));
    viz->view_settings_.camera_distance *= (1.0f - yoffset * 0.1f);
    viz->view_settings_.camera_distance = std::max(1.0f, std::min(20.0f, viz->view_settings_.camera_distance));
}

void RodVisualizer::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    RodVisualizer* viz = static_cast<RodVisualizer*>(glfwGetWindowUserPointer(window));
    
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_A:
                viz->view_settings_.show_axes = !viz->view_settings_.show_axes;
                break;
            case GLFW_KEY_C:
                viz->view_settings_.show_container = !viz->view_settings_.show_container;
                break;
            case GLFW_KEY_R:
                viz->anim_settings_.auto_rotate = !viz->anim_settings_.auto_rotate;
                break;
            case GLFW_KEY_S:
                viz->save_frame("screenshot.png");
                std::cout << "Screenshot saved as screenshot.png" << std::endl;
                break;
        }
    }
}

void RodVisualizer::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    RodVisualizer* viz = static_cast<RodVisualizer*>(glfwGetWindowUserPointer(window));
    viz->window_width_ = width;
    viz->window_height_ = height;
    glViewport(0, 0, width, height);
}

void RodVisualizer::window_close_callback(GLFWwindow* window) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}

// OptimizationRecorder implementation
OptimizationRecorder::OptimizationRecorder(const Settings& settings)
    : settings_(settings), visualizer_(1024, 768, "Optimization Recording"), frame_count_(0) {
    
    // Create output directory
    std::filesystem::create_directories(settings_.output_dir);
}

OptimizationRecorder::OptimizationRecorder()
    : OptimizationRecorder(Settings{}) {
}

void OptimizationRecorder::finalize() {
    if (settings_.create_video && frame_count_ > 0) {
        // Create FFmpeg command to generate video
        std::string cmd = "ffmpeg -y -r " + std::to_string(settings_.fps) + 
                         " -i " + settings_.output_dir + "/frame_%d.png" +
                         " -c:v libx264 -pix_fmt yuv420p " + settings_.video_filename;
        
        std::cout << "Creating video: " << cmd << std::endl;
        int result = system(cmd.c_str());
        
        if (result == 0) {
            std::cout << "Video created: " << settings_.video_filename << std::endl;
            
            if (!settings_.keep_frames) {
                // Clean up frame files
                for (int i = 0; i < frame_count_; ++i) {
                    std::string frame_file = settings_.output_dir + "/frame_" + std::to_string(i) + ".png";
                    std::filesystem::remove(frame_file);
                }
                std::filesystem::remove(settings_.output_dir);
            }
        } else {
            std::cerr << "Failed to create video. FFmpeg may not be installed." << std::endl;
        }
    }
}

} // namespace entanglement