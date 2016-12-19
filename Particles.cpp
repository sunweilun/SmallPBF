/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Particles.cpp
 * Author: swl
 * 
 * Created on April 15, 2016, 12:16 PM
 */

#include "Particles.h"

double Particles::lambda(unsigned i) const
{
    const std::vector<unsigned>& neighbors = neighborsList[i];
    
    double sum_nabla_sqr = 0.0;
    
    glm::dvec3 self_nabla(0);
    
    double density = 0.0;
    
    for(const unsigned &ni : neighbors)
    {
        density += w_poly6(particles[ni].p - particles[i].p, kernel_size);
        if(ni == i)
            continue;
        glm::dvec3 dp = particles[i].p - particles[ni].p;
        glm::dvec3 nabla = w_spiky_gradient(dp, kernel_size) / rest_density;
        self_nabla += nabla;
        sum_nabla_sqr += glm::dot(nabla, nabla);
    }
    sum_nabla_sqr += glm::dot(self_nabla, self_nabla);
    float lamb = -(density/rest_density-1) / (sum_nabla_sqr + epsilon);
    return lamb;
}
    
std::vector<unsigned> Particles::getNeighbors(unsigned index) const
{
    std::vector<unsigned> neighbors;
    HashCell c(particles[index].p, kernel_size);
    for(int off = 0; off<27; off++)
    {
        HashCell nc = c;
        nc.x += off%3-1;
        nc.y += (off/3)%3-1;
        nc.z += (off/9)%3-1;
        auto it = hashgrid.find(nc);
        if(it == hashgrid.end())
            continue;
        neighbors.insert(neighbors.end(), it->second.begin(), it->second.end());
    }
    return neighbors;
}

Particles::Particles() 
{
    int nx = 20;
    int ny = 40;
    int nz = 20;
    double d = 0.05;
    kernel_size = 0.06;
    radius = d*0.45;
    k = 0.01;
    n = 4;
    q = 0.2;
    epsilon = 1e5;
    rest_density = 1/(d*d*d);
    dt = 0.002;
    
    for(int x=0; x<nx; x++)
    {
        for(int y=0; y<ny; y++)
        {
            for(int z=0; z<nz; z++)
            {
                Particle par;
                par.p = glm::dvec3((x+0.5-nx*0.5)*d, (y+0.5)*d-1.0, (z+0.5-nz*0.5)*d);
                particles.push_back(par);
            }
        }
    }
}

void Particles::applyCollision(unsigned i)
{
    Particle& par = particles[i];
    par.p = glm::max(par.p, glm::dvec3(-1, -1, -1));
    par.p = glm::min(par.p, glm::dvec3(1, 1, 1));
}

void Particles::step()
{
    std::vector<double> lamb(particles.size());
    
    std::vector<glm::dvec3> old_positions(particles.size());
    for(unsigned i=0; i<particles.size(); i++)
    {
        old_positions[i] = particles[i].p;
    }
    
    for(Particle& par : particles)
    {
        par.v += glm::dvec3(-0.5, -9.8, 0) * dt;
        par.p += par.v * dt;
    }
    
    updateHashgrid();
    
    for(int iter = 0; iter < 10; iter ++)
    {
        #pragma omp parallel for schedule(dynamic, 128)
        for(int i=0; i<lamb.size(); i++)
        {
            lamb[i] = lambda(i);
        }
        #pragma omp parallel for schedule(dynamic, 128)
        for(int i=0; i<particles.size(); i++)
        {
            glm::dvec3 delta_p(0);
            const std::vector<unsigned>& neighbors = neighborsList[i];
            for(const unsigned& j : neighbors)
            {
                glm::dvec3 dp = particles[i].p - particles[j].p;
                double s_corr = -pow(w_poly6(dp, kernel_size)/w_poly6(glm::dvec3(q, 0, 0)*kernel_size, kernel_size), n)*k;
                delta_p += (lamb[i]+lamb[j]+s_corr)*w_spiky_gradient(dp, kernel_size);
            }
            delta_p /= rest_density;
            particles[i].p += delta_p;
            applyCollision(i);
        }
    }
    
    for(unsigned i=0; i<particles.size(); i++)
    {
        glm::dvec3 perturb;
        particles[i].v = (particles[i].p - old_positions[i]+perturb) / dt;
    }
}

void Particles::render() const
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    glEnable(GL_DEPTH_TEST);
    glutSolidCube(2.0);
    glPopAttrib();
    
    GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat mat_shininess[] = { 50.0 };
    GLfloat light_position[] = { 10.0, 10.0, 10.0, 0.0 };
    glShadeModel (GL_SMOOTH);
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_DIFFUSE);
    glColor3f(0.2, 0.5, 0.8);
    glColorMaterial(GL_FRONT, GL_SPECULAR);
    glColor3f(0.9, 0.9, 0.9);
    glColorMaterial(GL_FRONT, GL_AMBIENT);
    glColor3f(0.2, 0.5, 0.8);
    
    for(const Particle &par : particles)
    {    
        
        glPushMatrix();
        glTranslatef(par.p.x, par.p.y, par.p.z);
        glutSolidSphere(radius, 10, 10);
        glPopMatrix();
    }
    
    glPopAttrib();
}

void Particles::updateHashgrid()
{
    hashgrid.clear();
    for(unsigned i=0; i<particles.size(); i++)
    {
        hashgrid[HashCell(particles[i].p, kernel_size)].push_back(i);
    }
    neighborsList.clear();
    neighborsList.resize(particles.size());
    for(unsigned i=0; i<particles.size(); i++)
    {
        neighborsList[i] = getNeighbors(i);
    }
}

