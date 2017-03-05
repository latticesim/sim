/*
 * Simulation of clonal evolution for cells on a hexagonal lattice 
 * Called from sim.py script via ctypes 
 * 
 * Python entry points prefixed with py_
 *
 * M Lynch 2017
 */

#include <assert.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <random>
#include <fstream>
#include <memory>

#define R(v) v.begin(),v.end()
#define STR( x ) static_cast< std::ostringstream & >( ( std::ostringstream() << std::dec << x ) ).str()



using namespace std;


/*
 * Write vector to a binary file that can be loaded by numpy
 */

template<class T>
void save_vector(vector<T> &v,const char *filename)
{
    ofstream outfile(filename, ios::out|ios::binary);
    outfile.write((char*)&v[0], v.size() * sizeof(T));
}

template<class T>
void print_vector(vector<T> &v)
{
    for(auto x:v) cout << x << ' ';
}


inline minstd_rand rand_gen()
{
    random_device rd;
    minstd_rand gen(rd());
    return gen;
}

/* random integer */

inline int rand_int(int min,int max)
{
    minstd_rand gen = rand_gen();
    uniform_int_distribution<> randn(min,max);
    return randn(gen);
}

/* bernoulli distribution with prob p */

inline float rand_p(float p)
{
    assert(p>=0 && p<=1.0);
    auto gen = rand_gen();
    bernoulli_distribution dist(p); 
    return dist(gen);
}

/* choose 1 from n elements with equal probability */

template<class T>
inline T rand_choice(vector <T> &items)
{
    auto gen = rand_gen();
    uniform_int_distribution<> dist(0,items.size()-1);
    return items[dist(gen)];
}

/* choose 1 from n elements with probability specified by probs */

template<class T>
inline T rand_p_choice(vector<T> &items,vector<float> &probs)
{
    vector<float> sums;
    sums.resize(items.size());
    partial_sum(R(probs),sums.begin());
    uniform_real_distribution<> dist(0.0,sums.back());
    auto gen = rand_gen();
    float p = dist(gen);
    for(int i=0;i<sums.size();++i) {
        if(p <= sums[i]) return items[i];
    }
    assert(false);
}


struct Pos {
    int x,y,z;
    Pos(int _x,int _y,int _z=0) 
    {
        x=_x;
        y=_y;
        z=_z;
    }
};

/* Implementation of hexagonal lattice */

class Grid {
public:
    vector<int> cells; //integer indicates identity of each cell
    int n;
    int height;
    vector<Pos> neighbours_even;
    vector<Pos> neighbours_odd;
    enum neighbour_modes {flat,vert,equiv,all};
    neighbour_modes neighbour_mode = neighbour_modes::equiv;

    /* 
     * Create from new grid and fill with zero 
     * 
     * Note that compiler will also create a default copy constructor which creates 
     * a duplicate grid when passed a reference to existing grid.
     *
     * */

    Grid(int _n=20,int _height=1)
    {
        n=_n;
        height=_height;
        cells.resize(n*n*height); 
        //cout << "Grid " << n*n*height << '|';
        fill(R(cells),0);

        for(auto p:{Pos(-1,0),Pos(1,0),Pos(0,-1),Pos(1,-1),Pos(0,1),Pos(1,1)}) neighbours_even.push_back(p);
        for(auto p:{Pos(-1,0),Pos(1,0),Pos(-1,-1),Pos(0,-1),Pos(-1,1),Pos(0,1)}) neighbours_odd.push_back(p);
    }

    /* Get cell identity at x,y,z */
    int get(int x,int y,int z=0) { return cells[(z*n+y)*n + x]; }
    int get(Pos pos) { return get(pos.x,pos.y,pos.z); } 

    /* Set cell identity at x,y,z */
    void set(int x,int y,int z,int val) { cells[(z*n+y)*n + x] = val; }
    void set(int x,int y,int val) { set(x,y,0,val); }
    void set(Pos pos,int val) { set(pos.x,pos.y,pos.z,val); }

    /* 
     * Find neighbouring positions on the hexagonal lattice
     * Modes:
     * - flat: neighbours only on same level of 3D lattice (or 2D lattice)
     * - vert: as flat for bottom of lattice, but for levels above this only vertical
     * - equiv: all cells in vertical stack are spatially equivalent
     *
     */
    
    vector<Pos> neighbours(int x,int y,int z,bool ignore_empty=true)
    {
        //Neighbours on hexagonal lattice differ according to whether odd or even row
        assert(x<n && y<n && z<height);
        vector<Pos> a;
        if(y%2==0) a = neighbours_even; 
        else a = neighbours_odd; 
        vector<Pos> v;
        
        //Neighbours adjacent on lattice and vertical stacks adjacent 
        if(neighbour_mode == neighbour_modes::equiv || z==0) {
            for(Pos pos:a) {
                int nx=x+pos.x;
                int ny=y+pos.y;
                //check that candidate neighbouring cell is within bounds of simulation and that is not empty space (<0)
                if(nx<n && nx>=0 && ny<n && ny>=0) {
                    if(neighbour_mode == neighbour_modes::equiv) {
                        for(int iz=0;iz<height;++iz) { 
                            if(!ignore_empty || get(nx,ny,iz)>=0) v.push_back(Pos(nx,ny,iz));
                        }

                    }
                    else {
                        if(!ignore_empty || get(nx,ny,z)>=0) v.push_back(Pos(nx,ny,z));
                    }
                }
            }
        }

        //Neighbours up/down lattice
        if(height>1 && neighbour_mode != neighbour_modes::flat) {
            if(neighbour_mode == neighbour_modes::vert) {
                //immediate neighbours above and below
                if(z>0 && (!ignore_empty || get(x,y,z-1)>=0)) v.push_back(Pos(x,y,z-1));
                if(z<height-1 && (!ignore_empty || get(x,y,z+1)>=0)) v.push_back(Pos(x,y,z+1));
            }
            else if(neighbour_mode == neighbour_modes::equiv) {
                //all cells in this vertical stack
                for(int iz=0;iz<height;++iz) { 
                    if(iz==z) continue; //the index lattice position
                    if(!ignore_empty || get(x,y,iz)>=0) v.push_back(Pos(x,y,iz));
                }
            }
        }

        return v;
    }

    vector<Pos> neighbours(Pos pos) { return neighbours(pos.x,pos.y,pos.z); }
    

    /* 
     * Choose a neighbour at random including empty position 
     */ 

    Pos random_neighbour(Pos pos)
    {
        //Pick neighbour at random
        random_device rd;
        vector<Pos> N = neighbours(pos.x,pos.y,pos.z,false);
        if(N.size()==0) return Pos(-1,-1,-1);
        Pos npos = rand_choice(N);
        return npos;

    }

    /* Swap cells on grid */

    void swap(Pos &pos1,Pos &pos2)
    {
        int temp = get(pos1.x,pos1.y,pos1.z);
        set(pos1.x,pos1.y,pos1.z,get(pos2.x,pos2.y,pos2.z));
        set(pos2.x,pos2.y,pos2.z,temp);
    }


    /*
     * Convert position in hexagonal lattice to position as plotted on 
     * two dimension square lattice 
     */

    static Pos hex_to_square(float hx,float hy)
    {
        float x = hx+0.5;
        float y = hy+0.5;
        if((int)hy % 2 == 0) x += 0.5;
        return Pos(x,y);
    }

    /*
     * Distance of centre of one cell to outer border of another
     */ 

    static float dist(float x1,float y1,float x2,float y2)
    {
        Pos pos1 = Grid::hex_to_square(x1,y1);
        Pos pos2 = Grid::hex_to_square(x2,y2);
        
        float x = abs(pos1.x-pos2.x);
        float y = abs(pos1.y-pos2.y);
        return sqrt(x*x + y*y)+0.5; //add 0.5 to reach outer border of circle
    }

    /*
     * Draw filled circle on hexagonal lattice 
     * Circle defined as all cells on the lattice completely contaned
     * within radius r
     */

    void draw_circle(int val,float xpos,float ypos,float r)
    {

        for(float x=xpos-r;x<xpos+r;x+=1) {
            for(float y=ypos-r;y<ypos+r;y+=1) {
                if(Grid::dist(x,y,xpos,ypos) < r) set(x,y,val);
            }
        }
    }

    /* 109  199
     *
     * 100  190
     *
     * Display grid in rectangular format with 0,0 at bottom left
     * Only works for 2D grid, for 3d grid use python
     */
    
    friend ostream& operator<< (ostream &out, Grid &grid) 
    {
        assert(grid.height ==1); //not tested for 3D grid

        int maxc = STR(*max_element(R(grid.cells))).length();
        
        for(int iy=grid.n-1;iy>=0;--iy) {
            out << '|';
            for(int ix=0;ix<grid.n;++ix) {
                int a = grid.get(ix,iy);
                string s;
                if(a>=0) s = STR(a);
                else s = '.';
                int l = s.length();
                if(l < maxc) {
                    for(int i=0;i<maxc-l;++i) s += ' '; 
                }
                out << s << '|';
            }
            out << '\n';
        }
        return out;
    }
};


/*
 * Combine several Grid to represent:
 * (i) cell identity (ii) differentiation stauts (iii) number of reps
 */

class MultiGrid {
public:
    Grid id; //Clonal identity
    Grid diff; //Differentiation status
    Grid reps; //Number of repliations in this diff status

    MultiGrid(int size=20,int height=1)
    {
        Grid _id(size,height);
        id=_id;
        Grid _diff(size,height);
        diff = _diff;
        Grid _reps(size,height);
        reps = _reps;
    }

    void set(Pos pos,int _id,int _diff,int _reps) 
    {
        id.set(pos,_id);
        diff.set(pos,_diff);
        reps.set(pos,_reps);
    }

    void swap(Pos pos1,Pos pos2)
    {
        id.swap(pos1,pos2);
        diff.swap(pos1,pos2);
        reps.swap(pos1,pos2);
    }
    
    void remove(Pos pos)
    {
        set(pos,-1,-1,-1);
    }

    void copy_pos(MultiGrid &src,Pos srcpos,Pos destpos)
    {
        id.set(destpos,src.id.get(srcpos));
        diff.set(destpos,src.diff.get(srcpos));
        reps.set(destpos,src.reps.get(srcpos));
    }
};



/*
 * Node is a branchpoint in clonal history of population
 * Stored within Tree
 */

class Node {
public:
    vector<int> children;
    int name=-1;
    //float p_rep=1.0;
    float p_die=1.0;
    int stem_mutant = 0; //does stem cell have mutation that prevents asymmetrical division?

    friend ostream& operator<< (ostream &out, Node &node) 
    {
        out << node.name << "->";
        for(auto &child:node.children) out << child << ' ';
        return out;
    }
 
};

/*
 * Tree stores the complete clonal history of population
 */

class Tree {
public: 
    int newest; //the name of the most recent node created
    map<int,Node> nodes;
    
    struct Iter {
        vector<int> nstack;
        set<int> marked;
        Tree *owner;
        
        void start() 
        {
            nstack.clear();  
            nstack.push_back(0); //founder clone - from which all other clones derived
            marked.clear();
        }
       
        Node *next()
        {
            while(!nstack.empty()) {
                int name = nstack.back(); 
                auto search = owner->nodes.find(name);
                assert(search != owner->nodes.end());
                Node &node = search->second;
                 
                if(!node.children.empty() && (marked.find(node.name) == marked.end())) {
                    for(auto &child:node.children) nstack.push_back(child); 
                    marked.insert(node.name);
                }
                else {
                    nstack.pop_back();
                    return &node;
                } 
            }
            //iteration completed
            return NULL;
        }
    } iter;
    
    void reset()
    {
        Node node;
        newest = 0;
        node.name = 0;
        nodes[0] = node;

    }

    Tree() 
    {
        iter.owner = this;
        reset();
    }

   
    int add(int parent,Node &child)
    {
        newest += 1;
        child.name = newest; 
        //nodes.push_back(child);
        nodes[newest] = child;
        nodes[parent].children.push_back(newest);
        return newest; 
    }
};


/*
 * Model evolution of clone on hexagonal lattice
 * neutral: neutral competition
 * two: two clones with different probabilities of replication
 * multi: multiple clones with inherited random probs of replication
 *
 */

class Model {
public:
    //Some Default model parameters
    
    //********************************************************** 
    //These variables are accessed directly from python to 
    //set model parameters. c_types specification in sim.py
    //must reflect this. If the definition of these parameters
    //is changed then need to update sim.py to reflect this.
    float p_mut = 0.05; //rate of new detectable mutations (neutral / non-neutral)
    float p_diff = 0.5/7.0; //rate of stem cell loss/replacement 
    float p_diff_ta = 0.5; //rate of cell loss TA cells
    float rep_hi = 2.0;
    float rep_lo = 1.0;
    float rep_mut = 0.1;
    float die_hi = 1.0;
    float die_lo = 0.5;
    float die_mut = 0.1; //Fraction of detectable mutations that are non-neutral
    float stem_mut_rate = 0.0;
    float multi_mut = 0.01;
    float multi_mut_mean = 1.0;
    float multi_mut_sd = 0.5;
    float migration_stem = 0.0;
    float migration_ta = 0.0;
    float niche_radius = 5.0;
    float niche_sep = 30.0;
    int ta_max_reps = 3;
    int callback_freq = 100;
    //***********************************************************
    
    enum modes {neutral,rep,die,multi,stem_ta};
    modes mode = modes::neutral; 
    MultiGrid grid; //clonal identity, diff status 
    Grid niche; //positional architcture of stem cell niche (constant)
    Tree tree; //complete clonal history
    void (*callback)(int _generation,Model *_model);

    void reset(int _size,int height=1)
    {
        MultiGrid g(_size,height);
        grid = g;
        tree.reset(); 
        callback_freq = -1;
        assert(grid.id.n==_size);

    }
    
    /* contructor */
    
    Model(int _size=100) { reset(_size); } 

    /* Return the node in tree corresponding to lattice position */

    void init_stem_ta() 
    {
        //Stem cell:name=0,type=0
        //TA cell:name=1,parent=0,type=1
        //Arrange stem cells according to template
        assert(grid.id.height==1); //stem ta will not yet work with 3d grid
        mode = modes::stem_ta;
        niche = niche_template(grid.diff.n,niche_sep,niche_radius);
        assert(niche.n == grid.diff.n);
        copy(R(niche.cells),grid.diff.cells.begin());
        //replace 1 in niche template with empty space
        replace(R(grid.diff.cells),1,-1); 
        copy(R(grid.diff.cells),grid.id.cells.begin()); 
    }
   


    /*
     * Create a 2D hexagonal lattice which provides a template for stem cell niche i.e. integrin bright areas at top of rete ridges. Stem cell niche denoted by 0, non-stem cell area by >0.
     */
     
    static Grid niche_template(int size=200,float sep=50,float r=5)
    {
        Grid niche(size);
        int xmax=size; int ymax=size; 
        fill(R(niche.cells),1);

        //Midpoint of circles
        int i =0;
        for(float y=sep/2;y<=ymax;y+=sep) {
            float xmin = sep/2;
            if(i%2 == 0) xmin += sep/2;
            for(float x=xmin;x<xmax;x+=sep) { 
                niche.draw_circle(0,x,y,r);
            }
            ++i;
        }
        return niche;
    }

    /*
     * Calculate energy of configuration of cells on the lattice
     * Stem cells want to associate with stem cells
     * TA cells want to associate with TA cells
     * Stem cells want to associate with their niche 
     * Compare lattice energy between two states to determine if cell 
     * should be swapped during migration
     *
     * Note: this function does not calculate an _absolute_ energy
     * only the components that can potentially be changed by moving
     * the central cell.
     *
     * @diff: differentiation status of all cells in lattice
     * @pos: position within the lattice
     * @cell_type: differentiation status of central cell
     */

    float lattice_energy(Grid &diff,int cell_type,Pos pos)
    {
        //int x=pos.x; int y=pos.y;
        //Is central cell stem or non-stem
        //int isstem = 0;
        //jif(diff.get(pos)==0) isstem=1;
        
        //Count stem and non-stem neighbours
        vector<Pos> N = diff.neighbours(pos);
        int stem=0,ta=0,empty=0;
        for(Pos &n: N) {
            int cell_type = diff.get(n);
            if(cell_type==0) stem +=1;
            else if(cell_type>0) ta += 1;
            else empty+=1;
        }
        assert(ta+stem == N.size());

        //Is central cell a stem cell niche?
        int isniche = 0;
        if(niche.get(pos)==0) isniche=1;
      
        float e = 0; 
        float penalty = 10;
        if(cell_type==0) {  //stem
            e = ta; 
            if(!isniche) e += penalty;
        }
        else { //TA
            e = stem - empty;
            if(isniche) e += penalty;
        }
        return e;
        
        //cout << stem << ' ' << ta << ' ' << isstem << ' ' << isniche << '\n';
    }

    /*
     * Compare lattice energy between two states
     * @diff: state of differentiation at each point on lattice
     */

    float compare_energy(Grid &diff,Pos pos1,Pos pos2)
    {
        int cell1 = diff.get(pos1); 
        int cell2 = diff.get(pos2); 
        float e1 = lattice_energy(diff,cell1,pos1);
        float e2 = lattice_energy(diff,cell2,pos2);
        float e3 = lattice_energy(diff,cell1,pos2);
        float e4 = lattice_energy(diff,cell2,pos1);
        
        return  (e3+e4) - (e1+e2);
    }
 
    /*
     * Stochastically introduce neutral or functional mutations
     */ 
  
    void mutate()
    {
        //Mutate cells 
        //Add detectable label can either be neutral or functional mutation
        //minstd_rand gen = rand_gen();
        //uniform_int_distribution<> randn(0,grid.id.n-1);

        float a = float(grid.id.n*grid.id.n*grid.id.height)*p_mut;
        for(int i=0;i<a;++i) {
            int x = rand_int(0,grid.id.n-1);
            int y = rand_int(0,grid.id.n-1);
            int z = rand_int(0,grid.id.height-1);

            Pos pos(x,y,z);
            int b = grid.id.get(x,y,z); //current identity at this lattice point
            if(b<0) continue; //empty lattice
            Node &parent = tree.nodes[b];
            
            assert(tree.nodes.find(b) != tree.nodes.end());
            int c = -1;
            Node node;
            //node.p_rep = parent.p_rep;
            node.p_die = parent.p_die;
            node.stem_mutant = parent.stem_mutant;

            switch(mode) {
                case modes::neutral: {
                    break; 
                }
                case modes::stem_ta: {
                    //Stochastically create stem cells that do not differentiate when leave the niche
                    if(rand_p(stem_mut_rate) && grid.diff.get(pos)==0) {
                        node.stem_mutant = 1;
                        node.p_die = die_lo;
                    }
                    break;
                }

                case modes::die: {
                    //Stochastically convert from normal to low prob of loss 
                    float p=die_hi;
                    if(rand_p(die_mut)) p = die_lo; 
                    node.p_die = min(parent.p_die,p);
                    break;
                }
            }
            //Create the mutation
            c = tree.add(b,node); //parent id, child
            grid.id.set(x,y,c);
        }
    }

    /*
     * Stochastically remove cells from the lattice
     */

    void remove()
    {
        //Remove cells by differentiation / death
        for(int iz=0;iz<grid.id.height;++iz) {
            for(int iy=0;iy<grid.id.n;++iy) {
                for(int ix=0;ix<grid.id.n;++ix) {
                    Pos pos(ix,iy,iz);
                    Node &clone = tree.nodes[grid.id.get(pos)];
                    float p_adj = p_diff * clone.p_die;
                    if(clone.stem_mutant) assert(clone.p_die == die_lo);

                    if(mode==modes::stem_ta ) {
                        if(rand_p(p_adj) && grid.diff.get(pos)==0) grid.remove(pos); //stem
                        if(rand_p(p_diff_ta) && grid.diff.get(pos)>0) grid.remove(pos); //ta
                    }
                    else { 
                        if(rand_p(p_adj)) grid.remove(pos);
                    }
                }
            }
        }
    }
   
    
    /* 
     * What is probability of a TA cell continuing self-renewal
     * divisions given the number of divisions completed?
     */

    float p_rep_ta(MultiGrid &grid, Pos &pos)
    {
        int reps = grid.reps.get(pos);
        if(reps>=ta_max_reps) return 0.00;
        else return 1.0;

    }
   
    /* 
     * Fill in gaps created with neighbouring cells 
     * Limit number of TA replications to a maximum value
     * Stem cells differentiate to TA if divide outside a niche
     * Need to iterate through gaps in random order otherwise there is a bias
     * towards the corner that iteration starts from with regard to TA replication
     * Replication occcurs by copying cells from grid -> next 
     */
    
    void replicate()
    {
        MultiGrid next(grid.id.n,grid.id.height);

        //First identify where all the gaps are
        //and copy the non-gaps into next
        vector<Pos> gaps;
        for(int iz=0;iz<grid.id.height;++iz) {
            for(int iy=0;iy<grid.id.n;++iy) {
                for(int ix=0;ix<grid.id.n;++ix) {
                    Pos pos(ix,iy,iz); 
                    if(grid.id.get(pos)<0) gaps.push_back(pos);
                    else next.copy_pos(grid,pos,pos); 
                }
            }
        }

        //Randomize order of gaps
        auto gen = rand_gen();
        shuffle(R(gaps),gen);
        
        //Iterate through gaps and replicate neighbours where possible
        for(Pos pos:gaps) {
            vector<Pos> N = grid.id.neighbours(pos);
            Pos src(-10000,-10000); //where to replicate cell from
            if(mode==modes::stem_ta) {
                //limit proliferation of TA cells
                vector<Pos> V;
                for(Pos n:N) {
                    assert(grid.diff.get(n)>=0);
                    if(grid.diff.get(n)==0 || grid.reps.get(n)<ta_max_reps) V.push_back(n);
                }
                N=V;

            }
            if(N.size()>0) { 
                src = rand_choice(N);
                //Update number of replications
                int newreps = grid.reps.get(src)+1;
                grid.reps.set(src,newreps);
                next.copy_pos(grid,src,pos); 
                next.reps.set(pos,newreps);
                
                //Differentiate non-mutant stem cells that divide outside a niche
                if(mode==modes::stem_ta && !tree.nodes[next.id.get(pos)].stem_mutant && niche.get(pos)>0) { 
                    assert(grid.diff.get(src)==next.diff.get(pos));
                    if(grid.diff.get(src)==0) {
                        next.diff.set(src,1); 
                        next.diff.set(pos,1); 
                        next.reps.set(src,0); 
                        next.reps.set(pos,0); 
                    }
                }
            }
            else next.remove(pos); //leave position empty
        }
        
        //Update grid with changes
        grid = next;
 
    }


    void migrate()
    {
        //Migrate cells by stochastically swapping neighbours
        //minstd_rand gen = rand_gen();

        if(migration_stem>0.0 || migration_ta>0.0) {
            for(int iz=0;iz<grid.id.height;++iz) {
                for(int iy=0;iy<grid.id.n;++iy) {
                    for(int ix=0;ix<grid.id.n;++ix) {
                        Pos pos1(ix,iy,iz); //current position in lattice
                        float p;
                        if(grid.diff.get(pos1) == 0) p = migration_stem;
                        else p = migration_ta;
                        
                        if(rand_p(p)) { 
                            //Pick neighbour at random
                            Pos pos2 = grid.id.random_neighbour(pos1);

                            //Swap if lattice energy lower or the same
                            if(pos2.x!=-1) { //check if has neighbours
                                float e = compare_energy(grid.diff,pos1,pos2); 
                                if(e<=0) grid.swap(pos1,pos2); 
                            }
                        }
                    }
                }
            }
        }
    }
 
  
    /* 
     * This is the main simulation loop
     */
   
    void iterate(int iterations) 
    {

        //Set up rand number generation
        //random_device rd;
        //minstd_rand gen(rd());

        for(int generation=0;generation<iterations;++generation) {
            
            //Stochastically mutate
            mutate();

            //Remove cells and replicate neighbours
            remove();
            
            //Replicate cells
            replicate();
             
            //Implement cell migration 
            migrate();
            
            //Remove clones that have been extinguished 
            if(generation % 100==0) prune_tree(); 
            
            if(callback_freq>0 && (generation % callback_freq==0)) callback(generation,this); 
            
        }
    }

    /*
     * Count the number of cells of each clone
     * Include cells on the grid and their descendents
     */

    map<int,int> clone_counts(float p_die=-1)
    {
        map<int,int> counts;
        
        //quantify the clones that are present on the grid at the moment
        for(auto i:grid.id.cells) {
            if(p_die != -1 && tree.nodes[i].p_die != p_die) continue;
            if(counts.find(i)==counts.end()) counts[i]=0;
            counts[i] += 1;
        }
        
        //add children to parents
        tree.iter.start();
        while(Node *node=tree.iter.next()) {
            for(auto &child:node->children) {
                if(p_die != -1 && tree.nodes[child].p_die != p_die) continue;
                if(counts.find(node->name)==counts.end()) counts[node->name]=0;
                counts[node->name] += counts[child]; 
            }
        }
        
        if(p_die==-1) {
            int empty = count(R(grid.id.cells),-1);
            assert(counts[0] == grid.id.cells.size() - empty);
        }
        return counts;
    }
    
    
    /*
     * For each independent mutation defined in tree, quantify the size of the clone
     * for this identity and all descendents. The iter method of tree is
     * critical to this as it ensures that all children have been quantified before
     * their parents are encountered in the iteration
     *
     * @p_rep: if != -1 then only return the clones with this p_rep
     * Note that we are adding children to parents if the child has the mutation
     * this mimics sequencing which measures clone size by unique mutations but 
     * that mutation may not be the cause of clonal expansion 
     */
    
    vector<float> clone_sizes(float p_die=-1)
    {
        auto counts = clone_counts(p_die); 

        vector<float> clones; 
        float total = grid.id.n*grid.id.n;
        for(auto i=counts.begin();i!=counts.end();++i) {
            if(i->first == 0) continue;
            if(i->second>0) clones.push_back((float)i->second / total);
        }
        
        return clones;
    }

    /*
     * Remove clones that have been extinguished 
     * i.e. not on the lattice and none of their descendents on the lattice
     * This is required as otherwise the list of clones becomes to large for memory 
     */

    void prune_tree()
    {
        auto counts = clone_counts(); 
        map<int,Node> surviving;
        
        //Delete nodes for which clones have no surviving cells 
        for(auto i=counts.begin();i!=counts.end();++i) {
            if(i->second > 0) {
                //Delete children when the relevant nodes no longer exist
                Node &node = tree.nodes[i->first];
                vector<int> children;
                for(auto child=node.children.begin();child!=node.children.end();++child) {
                    if(counts.find(*child) != counts.end() && counts[*child] > 0) children.push_back(*child);
                }
                node.children = children;
                surviving[i->first] = node;
            }
        }
        tree.nodes = surviving;
    }


    
    /* 
     * Create a vector containing the probability of deletion from stem cell compartment each cell at that position
     */
    
    vector <float>dump_p_die()
    {
        vector<float> v;
        for(auto c:grid.id.cells) v.push_back(tree.nodes[c].p_die);
        return v;
    }
    
    /* 
     * Create a vector containing whether a stem cell is mutant or not 
     */
    
    vector <int>dump_stem_mutant()
    {
        vector<int> v;
        for(int c:grid.id.cells) v.push_back(tree.nodes[c].stem_mutant);
        return v;
    }
    
    
    /* 
     * Kill all of the stem cells to observe what happens
     */
    
    void kill_stem()
    {
        for(int x=0;x<grid.id.n;++x) {
            for(int y=0;y<grid.id.n;++y) {
                Pos pos(x,y);
                if(grid.diff.get(pos)==0) grid.remove(pos);
            }
        }
    }

};


/*
 * Correct answers: 0=36, 1=8, 3=3, 4=4, 2=28, 5=13,8=8, 6=6, 7=7
 */

void test_tree()
{
    Tree tree;
    int a[] = {0,0,1,1,2,2,2,5};
    std::vector<int> v(a, a + sizeof(a) / sizeof(int) );
    for(auto &x:v) {
        Node node;
        tree.add(x,node);
    }
    vector<int> total;
    total.resize(tree.nodes.size());

    tree.iter.start();
    while(Node *node=tree.iter.next()) {
        total[node->name] = node->name;
        for(auto &child:node->children) {
            //total[node->name] += total[tree.nodes[child].name]; 
            total[node->name] += total[child]; 
        }
        cout << node->name << '=' << total[node->name] << '\n';
    }
}

void test_grid()
{
    Grid grid(10);
    grid.set(0,0,100);
    grid.set(0,9,109);
    grid.set(9,9,199);
    grid.set(5,5,-1);
    grid.set(9,0,190);
    cout << grid;
}

void test_neighbours()
{
    Grid grid(20);
    Pos v[] = {Pos(0,0),Pos(19,19),Pos(4,5),Pos(0,10),Pos(10,19),Pos(0,5),Pos(19,4),Pos(0,19),Pos(8,0),Pos(15,15)};
    for(auto p:v) grid.set(p.x,p.y,1);

    for(auto p:v) {
        vector<Pos> a = grid.neighbours(p.x,p.y,0);
        for(auto b:a) grid.set(b.x,b.y,0,2);
    }
    cout << grid;
}


/*=============================================================================*/
/* Python module interface - uses ctypes */

/* 
 * Following functions used to pass variable size arrays to python via ctypes 
 * ***Note that Python script is responsible for calling free_x_buffer to free memory*** 
 */

#define BUFFER(name,type) \
vector<vector<type> > name; \
extern "C" int get_##name##_size(int n) { return name[n].size(); } \
extern "C" type* get_##name(int n) { &name[n][0]; } \
extern "C" int free_##name(int n) { vector<type>().swap(name[n]); } \

BUFFER(int_buffers,int)
BUFFER(float_buffers,float)

void (*py_callback)(int generation);

/*
 * py_model can be accessed via ctypes to change model parameters
 * do not declare as extern "C"
 */

Model py_model;

extern "C" void py_create_model(int size)
{
    py_model.reset(size,1);
}

extern "C" void py_create_model_3d(int size,int height)
{
    //cout << "py_create_model " << size << ' ' << height << '|';
    py_model.reset(size,height);
}

void py_model_neutral_callback(int generation, Model *model)
{
    if(generation==0) return;
    int_buffers[0] = model->grid.id.cells;
    auto clones = model->clone_sizes();
    float_buffers[0] = clones;
    if(py_callback != NULL) py_callback(generation); 
}


extern "C" void py_model_neutral(int generations,void (*callback)(int generation)=NULL)
{
    py_callback = callback;
    float_buffers.resize(1);
    int_buffers.resize(1);
    py_model.mode = Model::modes::neutral;
    py_model.callback = py_model_neutral_callback;
    py_model.iterate(generations);
    auto clones = py_model.clone_sizes();
    float_buffers[0]=clones;
    int_buffers[0] = py_model.grid.id.cells;
     
}

void py_model_die_callback(int generation, Model *model)
{
    //if(generation==0) return;
    float_buffers.resize(4);
    float_buffers[0] = py_model.clone_sizes(py_model.die_hi);
    float_buffers[1] = py_model.clone_sizes(py_model.die_lo);
    float_buffers[2] = py_model.clone_sizes();
    float_buffers[3] = model->dump_p_die(); 
    int_buffers.resize(1);
    int_buffers[0] = model->grid.id.cells;
    if(py_callback != NULL) py_callback(generation); 
}

extern "C" void py_model_die(int generations,void (*callback)(int generation)=NULL)
{
    py_callback = callback;
    py_model.mode = Model::modes::die;
    py_model.callback = py_model_die_callback;
    py_model.iterate(generations);
    //call the callback to quantify clones for python to use
    py_model_die_callback(generations,&py_model);
}

void py_model_3d_callback(int generation, Model *model)
{
    //if(generation==0) return;
    float_buffers.resize(4);
    float_buffers[0] = py_model.clone_sizes(py_model.die_hi);
    float_buffers[1] = py_model.clone_sizes(py_model.die_lo);
    float_buffers[2] = py_model.clone_sizes();
    float_buffers[3] = model->dump_p_die(); 
    int_buffers.resize(1);
    int_buffers[0] = model->grid.id.cells;
    if(py_callback != NULL) py_callback(generation); 
}

extern "C" void py_model_3d(int generations,void (*callback)(int generation)=NULL)
{
    py_callback = callback;
    py_model.mode = Model::modes::die;
    //py_model.reset(50);
    py_model.callback = py_model_3d_callback;
    py_model.iterate(generations);
    //call the callback to quantify clones for python to use
    py_model_3d_callback(generations,&py_model);
}



void py_model_stem_ta_callback(int generation, Model *model)
{
    int_buffers.resize(4);
    int_buffers[0] = model->grid.id.cells;
    int_buffers[1] = model->grid.diff.cells;
    int_buffers[2] = model->grid.reps.cells;
    int_buffers[3] = model->dump_stem_mutant();
    float_buffers.resize(4);
    float_buffers[0] = py_model.clone_sizes(py_model.die_hi);
    float_buffers[1] = py_model.clone_sizes(py_model.die_lo);
    float_buffers[2] = py_model.clone_sizes();
    float_buffers[3] = model->dump_p_die(); 
    if(py_callback != NULL && generation>=0) py_callback(generation); 

}


extern "C" void py_model_stem_ta(int generations,void (*callback)(int generation)=NULL)
{
    py_callback = callback;
    py_model.init_stem_ta();
    py_model.callback = py_model_stem_ta_callback;
    
    py_model_stem_ta_callback(0,&py_model);
    py_model.iterate(generations);
    py_model_stem_ta_callback(-1,&py_model);
    
}

extern "C" void py_kill_stem()
{
    py_model.kill_stem();
}

extern "C" void test()
{
    py_create_model_3d(50,2);
    cout << "created...";
    py_model_3d(100);
    cout << "finished...";

}

int main( int argc, char** argv )
{
    //test_tree();
    //run_model_neutral();
    //py_model_neutral(100,100);
    //printtest(2,4.3);
    //test_clone_sizes();
    //py_model_stem_ta(1000);
    test();
    return 0;
}


 
