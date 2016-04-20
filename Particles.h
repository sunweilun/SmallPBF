/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Particles.h
 * Author: swl
 *
 * Created on April 15, 2016, 12:16 PM
 */

#ifndef PARTICLES_H
#define PARTICLES_H

#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#if defined(__APPLE_CC__)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#include <math.h>
#endif

struct HashCell
{
    long long int x;
    long long int y;
    long long int z;
    HashCell(const glm::dvec3& p, double size)
    {
        x = (long long int)(p.x / size);
        y = (long long int)(p.y / size);
        z = (long long int)(p.z / size);
    }
    bool operator==(const HashCell& t) const
    {
        return t.x == x && t.y == y && t.z == z;
    }
};

namespace std{
template<> 
struct hash<HashCell>
{
    size_t operator()(const HashCell& t) const
    {
        return t.x+t.y*15485967+t.z*452930477;
    }
};
}

class Particles {
public:
    Particles();
    void render() const;
    void step();
private:
    struct Particle
    {
        glm::dvec3 p;
        glm::dvec3 v;
    };
    
    double w_poly6(const glm::dvec3& diff_vec, double h) const
    {
        double r = glm::length(diff_vec);
        if(r > h)
            return 0;
        double h2 = h*h;
        double h4 = h2*h2;
        double h9 = h4*h4*h;
        double t = h2 - r*r;
        return 315/(64*M_PI*h9)*t*t*t;
    }
    
    glm::dvec3 w_spiky_gradient(const glm::dvec3& diff_vec, double h) const
    {
        double r = glm::length(diff_vec);
        if(r > h || r == 0)
            return glm::dvec3(0);
        double t = h - r;
        double h2 = h*h;
        double h6 = h2*h2*h2;
        return double(-45/(M_PI*h6)*t*t/r)*diff_vec;
    }
    
    double w_vis_laplacian(const glm::dvec3& diff_vec, double h) const
    {
        double r = glm::length(diff_vec);
        if(r > h)
            return 0;
        double h2 = h*h;
        double h6 = h2*h2*h2;
        return 45/(M_PI*h6)*(h-r);
    }
    
    double radius;
    double kernel_size;
    double epsilon;
    double dt;
    float k;
    float n;
    float q;
    
    void applyCollision(unsigned i);
    
    double lambda(unsigned i) const;
    
    std::vector<unsigned> getNeighbors(unsigned index) const;
    
    void updateHashgrid();
    
    double rest_density;
    std::vector<Particle> particles;
    std::unordered_map<HashCell, std::vector<unsigned> > hashgrid;
    std::vector<std::vector<unsigned> > neighborsList;
};

#endif /* PARTICLES_H */

