#include "ShaderUtils.h"


GLuint ShaderManager::CompileShader(GLuint type, const std::string& source)
{
    GLuint shaderObject;
    if (type == GL_VERTEX_SHADER)
    {
        shaderObject = glCreateShader(GL_VERTEX_SHADER);
    }
    else if (type == GL_FRAGMENT_SHADER)
    {
        shaderObject = glCreateShader(GL_FRAGMENT_SHADER);
    }

    const char* src = source.c_str();

    glShaderSource(shaderObject, 1, &src, nullptr);
    glCompileShader(shaderObject);

    GLint success;
    glGetShaderiv(shaderObject, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(shaderObject, 512, NULL, infoLog);
        std::cerr << "Shader compilation error:\n" << infoLog << std::endl;
    }

    return shaderObject;
}

std::string ShaderManager::LoadShaderFromFile(const std::string& fileName)
{
    std::ifstream inputStream(fileName.c_str(), std::ios::in);
    if (!inputStream.is_open())
    {
        throw std::runtime_error("File not found, or couldn't be open");
    }

    std::stringstream buffer;
    buffer << inputStream.rdbuf();

    inputStream.close();

    return buffer.str();
}

ShaderManager::ShaderManager(const std::string& vertexShaderFileName, const std::string& fragmentShaderFileName)
{
    std::string vertexShaderSource = LoadShaderFromFile(vertexShaderFileName);
    std::string fragmentShaderSource = LoadShaderFromFile(fragmentShaderFileName);

    programObject = glCreateProgram();

    vertexShader = CompileShader(GL_VERTEX_SHADER, vertexShaderSource);
    fragmentShader = CompileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    glAttachShader(programObject, vertexShader);
    glAttachShader(programObject, fragmentShader);
    glLinkProgram(programObject);

    glValidateProgram(programObject);
}

GLuint ShaderManager::GetProgramObject()
{
    return programObject;
}

GLuint ShaderManager::GetVertexShader()
{
    return vertexShader;
}

GLuint ShaderManager::GetFragmentShader()
{
    return fragmentShader;
}