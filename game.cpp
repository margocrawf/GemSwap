
#define USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


unsigned int windowWidth = 512, windowHeight = 512;

bool keyboardState[256];
int clicked[2];

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) 
{
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) 
	{
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) 
{
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) 
	{
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) 
{
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) 
	{
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = "\n\
	#version 130 \n\
	precision highp float; \n\
	in vec2 vertexPosition;	\n\
	out vec3 color; \n\
	uniform mat4 M; \n\
	uniform vec3 vertexColor; \n\
	void main() \n\
	{ \n\
		color = vertexColor; \n\
		gl_Position = vec4(vertexPosition.x, \n\
		vertexPosition.y, 0, 1) * M; \n\
	} \n\
"; 

// fragment shader in GLSL
const char *fragmentSource = "\n\
	#version 130 \n\
	precision highp float; \n\
	\n\
	in vec3 color;			// variable input: interpolated from the vertex colors \n\
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation \n\
	\n\
	void main() \n\
	{ \n\
		fragmentColor = vec4(color, 1); // extend RGB to RGBA \n\
	} \n\
";

// row-major matrix 4x4
struct mat4 
{
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) 
	{
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) 
	{
		mat4 result;
		for (int i = 0; i < 4; i++) 
		{
			for (int j = 0; j < 4; j++) 
			{
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 
{
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) 
	{
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) 
	{
		vec4 result;
		for (int j = 0; j < 4; j++) 
		{
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	vec4 operator+(const vec4& vec) 
	{
		vec4 result(v[0] + vec.v[0], v[1] + vec.v[1], v[2] + vec.v[2], v[3] + vec.v[3]);
		return result;
	}
};

// 2D point in Cartesian coordinates
struct vec2 
{
	float x, y;

	vec2(float x = 0.0, float y = 0.0) : x(x), y(y) {}

	vec2 operator+(const vec2& v) 
	{
		return vec2(x + v.x, y + v.y);
	}

	vec2 operator*(float s) 
	{
		return vec2(x * s, y * s);
	}
};


// shader ID

class Shader
{
unsigned int shaderProgram;

public:
    Shader()
    {
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }

        glShaderSource(vertexShader, 1, &vertexSource, NULL);
        glCompileShader(vertexShader);
        checkShader(vertexShader, "Vertex shader error");

        // create fragment shader from string
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }

        glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
        glCompileShader(fragmentShader);
        checkShader(fragmentShader, "Fragment shader error");

        // attach shaders to a single program
        shaderProgram = glCreateProgram();
        if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }

        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        // connect Attrib Array to input variables of the vertex shader
        glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0

        // connect the fragmentColor to the frame buffer memory
        glBindFragDataLocation(shaderProgram, 0, "fragmentColor"); // fragmentColor goes to the frame buffer memory

        // program packaging
        glLinkProgram(shaderProgram);
        checkLinking(shaderProgram);
    }

    void Run() {
        glUseProgram(shaderProgram);
    }

    void UploadColor(vec4& color) {
		int location = glGetUniformLocation(shaderProgram, "vertexColor");
		if (location >= 0) glUniform3fv(location, 1, &color.v[0]); 
		else printf("uniform vertexColor cannot be set in Shader class\n");
    }

    void UploadM(mat4& M) {
		int location = glGetUniformLocation(shaderProgram, "M");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, M); 
		else printf("uniform M cannot be set in Shader class\n");
    }

};


Shader* gShader;

class Material
{
    Shader* shader;
    vec4 color;

    public:
        Material(Shader* shad, vec4 c) {
            shader = shad;
            color = c;
    }

    void UploadAttributes() {
        shader->UploadColor(color);
    }

    Shader* getShader() {
        return shader;
    }

    /* if this works, make a heart material subclass */
    void UpdateColor(double t) {
        float shade = .3 * sin(t) + .7; //sin, but make it all positive
        color = vec4(shade, 0.0, 0.0);
        UploadAttributes();
    }

};

class Camera
{
	vec2 center;
	vec2 halfSize;

public:
	Camera()
	{
		center = vec2(0.0, 0.0);
		halfSize =  vec2(1.0, 1.0);
	}

	mat4 GetViewTransformationMatrix()
	{
		mat4 T = mat4(
			1.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			-center.x, -center.y, 0.0, 1.0);

		mat4 S = mat4(
			1.0 / halfSize.x, 0.0, 0.0, 0.0,
			0.0, 1.0 / halfSize.y, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 1.0);

		return T * S;
	}

	void SetAspectRatio(int width, int height)
	{
		halfSize = vec2((float)width / height,1.0);
	}

	void Move(float dt)
	{
		if(keyboardState['e']) center = center + vec2(1.0, 0.0) * dt;
		if(keyboardState['a']) center = center + vec2(-1.0, 0.0) * dt;
		if(keyboardState[',']) center = center + vec2(0.0, 1.0) * dt;
		if(keyboardState['o']) center = center + vec2(0.0, -1.0) * dt;
	}
};

Camera camera;

class Geometry
{
    protected: unsigned int vao;

    public:
        Geometry()
        {
            glGenVertexArrays(1, &vao);
        }

        // Draw function-- defined by subclasses
        virtual void Draw() = 0;

};

class Triangle : public Geometry
{
    unsigned int vbo;
public:	

    Triangle()
	{

        glBindVertexArray(vao);

		glGenBuffers(1, &vbo);		// generate a vertex buffer object

		// vertex coordinates: vbo -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		static float vertexCoords[] = { -.9, .865,
                                         .9, .865,
                                         0, -.865 };	// vertex data on the CPU

		glBufferData(GL_ARRAY_BUFFER,	// copy to the GPU
			sizeof(vertexCoords),	// size of the vbo in bytes
			vertexCoords,		// address of the data array on the CPU
			GL_STATIC_DRAW);	// copy to that part of the memory which is not modified 
		
		// map Attribute Array 0 to the currently bound vertex buffer (vbo)
		glEnableVertexAttribArray(0);
		
		// data organization of Attribute Array 0 
		glVertexAttribPointer(0,	// Attribute Array 0 
			2, GL_FLOAT,		// components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalize
			0, NULL);		// stride and offset: it is tightly packed	

	}
    
    void Draw()
    {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
};

class Quad : public Geometry
{
    unsigned int vbo;

public:
    Quad() {
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        static float vertexCoords[] = { -.8, -.8,
                                        -.8, .8,
                                        .8, -.8,
                                        .8, .8};
        
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW); 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Draw() {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); 
    }
};


class Hex : public Geometry
{
    unsigned int vbo;

public:
    Hex() {
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        static float vertexCoords[] = { 0, 0, 
                                        1, 0,
                                        0.5, 0.87,
                                        -0.5, 0.87,
                                        -1, 0,
                                        -0.5, -0.87,
                                        0.5, -0.87,
                                        1, 0};
        
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW); 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Draw() {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 8); 
    }
};

class Star : public Geometry
{
    unsigned int vbo;

public:
    Star() {
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        static float vertexCoords[] = { 0, 0,
                                        0, 1, // 90, 1
                                        -0.294, 0.405, // 126, .5
                                        -0.951, 0.309, // 162, 1
                                        -0.476, -0.155, // 198, .5
                                        -0.588, -0.809, // 234, 1
                                        0, -0.5, // 270, -.5
                                        0.588, -0.809,
                                        0.476, -0.155,
                                        0.951, 0.309,
                                        0.294, 0.405,
                                        0, 1 };

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW); 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    }

    void Draw() {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 12); 
    }
};

class Heart : public Geometry
{
    unsigned int vbo;

public:
    Heart() {
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        static float vertexCoords[] = { 0.0, 0.0,
                                        0.0, 0.31571636042179707, 
                                        0.0345853202883177, 0.4426684215281069, 
                                        0.2341011078901171, 0.676533673483712, 
                                        0.5927650046111893, 0.7431210050696208, 
                                        0.9203573172927, 0.5301695443576366, 
                                        0.9999756885671125, 0.15290060048563875, 
                                        0.7759094397095047, -0.23013751343618305, 
                                        0.40233857910708554, -0.5672591664559598, 
                                        0.10892408564868375, -0.858814512893784, 
                                        0.004504989026994553, -1.0470402323496864, 
                                        -0.004504989026994532, -1.0470402323496861, 
                                        -0.10892408564868364, -0.8588145128937842, 
                                        -0.4023385791070846, -0.5672591664559604, 
                                        -0.7759094397095041, -0.23013751343618377, 
                                        -0.9999756885671125, 0.15290060048563822, 
                                        -0.9203573172927003, 0.5301695443576362, 
                                        -0.59276500461119, 0.7431210050696206, 
                                        -0.23410110789011734, 0.6765336734837122, 
                                        -0.03458532028831779, 0.44266842152810715, 
                                        -1.484463788708891e-47, 0.31571636042179707 };

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW); 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }
    void Draw() {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 21);
    }
};


class Mesh
{
    Material* material;
    Geometry* geometry;
    std::string gemType;

    public:
        Mesh(Material* mat, Geometry* geo, std::string type="gem") {
            material = mat;
            geometry = geo;
            gemType = type;
        }
        
        void Draw() {
            material->UploadAttributes();
            geometry->Draw();
        }

        Shader* getShader() {
            return material->getShader();
        }

        std::string getGemType() {
            return gemType;
        }
};

class Object
{
    Shader* shader;
    Mesh* mesh;
    vec2 position;
    float orientation;

    public:
        vec2 scaling;
        vec2 index;
        std::string gemType;
        Object( Mesh* m, vec2 pos = vec2(0.0, 0.0), vec2 sca = vec2(1.0, 1.0),
                float ori=0.0, vec2 i = vec2(0,0)) {
            mesh = m;
            position = pos;
            scaling = sca;
            orientation = ori;
            index = i;

            gemType = mesh->getGemType();
            shader = mesh->getShader();
        }
        
        void UploadAttributes() {

            // calculate T, S, R from position, scaling, and orientation
            mat4 T = mat4(
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                position.x, position.y, 0.0, 1.0);

            mat4 S = mat4(
                scaling.x, 0.0, 0.0, 0.0,
                0.0, scaling.y, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 1.0);

            float alpha = orientation / 180.0 * M_PI;

            mat4 R = mat4(
                cos(alpha), sin(alpha), 0.0, 0.0,
                -sin(alpha), cos(alpha), 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0);
            
            mat4 V = camera.GetViewTransformationMatrix();

            mat4 M = S * R * T * V;

            shader->UploadM(M);
        }

        void Draw() {
            UploadAttributes();
            mesh->Draw();
        }

        void Move(vec2 dpos, vec2 dsca, float dori) {
            position.x += dpos.x;
            position.y += dpos.y;
            scaling.x += dsca.x;
            scaling.y += dsca.y;
            orientation += dori;
            Draw();
        }

        std::string getGemType() {
            return gemType;
        }

        vec2 getPos() {
            return position;
        }

};

class Grid {
    Shader* shader;
    std::vector<Mesh*> meshes;
    std::vector<std::vector<Object*>> objects;
    Material* heartMat;
    Material* starMat;
    Material* hexMat;
    Material* sqMat;
    Material* trMat;
    Material* voidMat;
    Mesh* voidMesh;
    float xscale;
    float yscale;
    vec2 selected;

public:

    Grid() { shader = 0; }

    /* takes scale, which is how big each object in the row is,
     * and ycoord, the y coordinate of the center of the row
     */
    void Initialize( int xcount, int ycount) {
        xscale = 1.0 / xcount;
        yscale = 1.0 / ycount;
        float xinit = -1.0 + xscale; // position of the first point
        float yinit = -1.0 + yscale;

        shader = new Shader();

        heartMat = new Material(shader, vec4(1,0,0));
        Mesh* heartMesh = new Mesh(heartMat, new Heart(), "heart");
        meshes.push_back(heartMesh);

        starMat = new Material(shader, vec4(1, 1, 0));
        meshes.push_back(new Mesh(starMat, new Star(), "star"));

        hexMat = new Material(shader, vec4(0,.4,.8));
        Mesh* hexMesh = new Mesh(hexMat, new Hex(), "hex");
        meshes.push_back(hexMesh);

        sqMat = new Material(shader, vec4(1, .5, 0));
        meshes.push_back(new Mesh(sqMat, new Quad(), "quad"));

        trMat = new Material(shader, vec4(.5,0,1));
        meshes.push_back(new Mesh(trMat, new Triangle(), "triangle"));

        voidMat = new Material(shader, vec4(0,0,0));
        voidMesh = new Mesh(voidMat, new Triangle(), "void");

        for (int j = 0; j < ycount; j++) {
           
            std::vector<Object*> objectsRow;

            float ycoord = yinit + j*yscale*2;

            for (int i = 0; i < xcount; i++ ) {
                
                float xcoord = xinit + i*xscale*2;

                Mesh* mesh = meshes[(rand() % meshes.size())];

                objectsRow.push_back(new Object(
                            mesh, vec2(xcoord, ycoord), vec2(xscale, yscale), 0.0, vec2(i,j)));

            }

            objects.push_back(objectsRow);
        }

        shader->Run();
    }

    ~Grid() {
        for(int i = 0; i < meshes.size(); i++) delete meshes[i];
        for(int i = 0; i < objects.size(); i++) { 
            for (int j = 0; j < objects[i].size(); j++ ) delete objects[i][j]; };
        if(shader) delete shader;
    }

    void Draw() {
        for ( int i=0; i < objects.size(); i++) { 
            for ( int j = 0; j < objects[i].size(); j++ ) objects[i][j]->Draw(); 
        };
    }

    void DeleteShape(int xInd, int yInd) {
        Object* o = objects[yInd][xInd];
        o->gemType = "exiting";
        /*
        objects[yInd][xInd] = new Object(voidMesh, o->getPos(), vec2(xscale, yscale), 0.0, o->index);
        delete o;
        */

    }

    void pulseHearts(double t, double dt) {
        heartMat->UpdateColor(t);
        shader->Run();
    }
    
    void Gyro(double t, double dt) {
        for ( int i = 0; i < objects.size(); i++) {
            for (int j = 0; j < objects[i].size(); j++ ) {
                Object* o = objects[i][j];
                if (o->getGemType() == "star") {
                    o->Move( vec2(0,0), vec2(0,0), dt*20.0);
                }
                if (o->getGemType() == "exiting") {
                    if ((o->scaling.x > 0.002) and (o->scaling.y > 0.002)) {
                        o->Move( vec2(0,0), vec2(-0.002, -0.002), dt*20);
                    } else {
                        objects[i][j] = new Object(voidMesh, o->getPos(), vec2(xscale, yscale), 0.0, o->index);
                        delete o;
                        Skyfall(j,i);
                    }
                }
            }
        }
    }
    void set_selected(vec2 sel) {
        selected = sel;
    }

    void swap_sub(vec2 sub) {
        swap(selected, sub);
    }

    void swap(vec2 sel, vec2 sub) {
        Object* selOb = objects[sel.y][sel.x];
        Object* subOb = objects[sub.y][sub.x];
        vec2 diff = vec2( (selOb->getPos().x - subOb->getPos().x), (selOb->getPos().y - subOb->getPos().y) );
        selOb->Move(vec2(-diff.x, -diff.y), vec2(0,0), 0.0);
        subOb->Move(diff, vec2(0,0), 0.0);
        objects[sel.y][sel.x] = subOb;
        objects[sub.y][sub.x] = selOb;
    }


    bool Legal(Object* selOb, Object* subOb) {
        return true;
    }

    void Skyfall(int col, int row) {
        for (int i = row; i < objects[row].size()-1; i++) {
            swap(vec2(col, i), vec2(col, i+1));
        }
        Object* orig = objects[objects.size()-1][col];
        Mesh* mesh = meshes[(rand() % meshes.size())];
        objects[objects.size()-1][col] = new Object(mesh, orig->getPos(), vec2(orig->scaling.x, orig->scaling.y), 0.0, vec2(objects.size()-1, col));
        delete orig;
    }

};


Material* gMaterial;
Geometry* gGeometry;
Mesh* gMesh;
Object* gObject;
Grid* gGrid;

// initialization, create an OpenGL context
void onInitialization() 
{
	for(int i = 0; i < 256; i++) keyboardState[i] = false;

	glViewport(0, 0, windowWidth, windowHeight);
 
    gGrid = new Grid();
    gGrid->Initialize( 10, 10 );

    
    
}

void onExit() 
{
    delete gMaterial;
    delete gGeometry;
    delete gMesh;
    delete gObject;
    delete gShader;
    delete gGrid;
	printf("exit");
}

// window has become invalid: redraw
void onDisplay() 
{	
	
	glClearColor(0, 0, 0, 0); // background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	
    /*
    gObject->Draw();
    */

    gGrid->Draw();
	glutSwapBuffers(); // exchange the two buffers
	
}

void onKeyboard(unsigned char key, int x, int y)
{
	keyboardState[key] = true;
}

void onKeyboardUp(unsigned char key, int x, int y)
{
	keyboardState[key] = false;
}

void onMouseMotion(int x, int y) {
}


void onMouse(int button, int state, int xOrig, int yOrig) {
    float x = 2.0 * ((float) xOrig / windowWidth - 0.5);
    float y = -2.0 * ((float) yOrig / windowHeight - 0.5);
    int xIndex = 5 * x + 5;
    int yIndex = 5 * y + 5;
    if ((button == GLUT_RIGHT_BUTTON) and (state == GLUT_DOWN)) {
        gGrid->DeleteShape(xIndex, yIndex);

    }
    else if ((button == GLUT_LEFT_BUTTON) and (state == GLUT_DOWN)) {
        gGrid->set_selected(vec2(xIndex,yIndex));
    }
    else if ((button == GLUT_LEFT_BUTTON) and (state == GLUT_UP)){
        gGrid->swap_sub(vec2(xIndex,yIndex));
    }
}

void onReshape(int winWidth0, int winHeight0) 
{
	camera.SetAspectRatio(winWidth0, winHeight0);
	glViewport(0, 0, winWidth0, winHeight0);
    windowWidth = winWidth0;
    windowHeight = winHeight0;
}

void onIdle( ) {
    // time elapsed since program started, in seconds
    double t = glutGet(GLUT_ELAPSED_TIME) * 0.001;
    // variable to remember last time idle was called
    static double lastTime = 0.0;
    // time difference between calls: time step  
    double dt = t - lastTime;
    // store time
    lastTime = t;

	camera.Move(dt);

    gGrid->pulseHearts(t, dt);

    gGrid->Gyro(t, dt);

    glutPostRedisplay();
}

int main(int argc, char * argv[]) 
{
    srand(time(NULL));
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight); 	// application window is initially of resolution 512x512
	glutInitWindowPosition(50, 50);			// relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow("Triangle Rendering");

#if !defined(__APPLE__)
    //printf("hello!");
	glewExperimental = true;	
	glewInit();
#endif
	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	
	onInitialization();

	glutDisplayFunc(onDisplay); // register event handlers
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutReshapeFunc(onReshape);
    glutMouseFunc(onMouse);

	glutMainLoop();
	onExit();
	return 1;
}


