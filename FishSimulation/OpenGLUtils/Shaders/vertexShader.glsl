#version 410 core
layout(location = 0) in vec2 vertexPosition;
layout(location = 1) in float posX;           // Instance-specific position X
layout(location = 2) in float posY;           // Instance-specific position Y
layout(location = 3) in float velX;           // Instance-specific velocity X
layout(location = 4) in float velY;           // Instance-specific velocity Y

uniform mat4 projection;

void main() {
    float angle = atan(-velY, velX);
    float c = cos(angle);
    float s = sin(angle);
    mat2 rotation = mat2(c, -s, s, c);

    vec2 rotatedPos = rotation * vertexPosition;
    vec2 finalPos = rotatedPos + vec2(posX, posY); 

    gl_Position = projection * vec4(finalPos, 0.0, 1.0);
}