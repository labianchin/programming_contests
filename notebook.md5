
## Libraries

	#include<iostream>
	#include<iomanip>
	#include<fstream>

## Makefile
	
	PROG = 11631

	${PROG}: ${PROG}.cpp
		g++ -Wall -lm -lcrypt -O2 -o ${PROG} ${PROG}.cpp

	test: 
		./${PROG} < in
		
	vagrind:	
		valgrind --tool=memcheck ./${PROG} < in
	
	all: ${PROG} test

## I/O

	using namespace std;

	std::getline(inf, jersey_number, '|'); #read thru pipe
	std::getline(inf, best_time);          #read thru newline

	std::cout.fill('X');
	std::cout << setw(2) << one << std::endl.;

	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	std::cout.setf(std::ios::showpoint);
	std::cout.precision(2);
	
	
	import sys

    for line in sys.stdin:
        print line
        int(line)
        map(lambda x: int(x), line.split(" "))

## Data Structures

* Vector, Lists, Sets, Stack, Queue, Priority_Queue, Heap, Map, MultiMap....
* *Vector*: push_back, pop_back, size, capacity, resize, front, back, clear	
* *queue*: front, back, push, pop
* *Algorithms*: sort, lower_bound, upper_bound, equal_range, binary_search, max, min, max_element, next_permutation, etc

### Bit Vec

	enum{ BITSPREAD = 32, SHIFT = 5, MASK = 0x1F};
	struct BitVec{
		int n, hi, *x;
		void set(int i){	x[i>>SHIFT] |= (1 << (i&&MASK)); }
		void clr(int i){	x[i>>SHIFT] &= (1 << (i&&MASK)); }
		void test(int i){return	x[i>>SHIFT] & (1 << (i&&MASK)); }
		void insert(int i){
			if (!test(i)){
				set(i);
				n++;
			}
		}
		BitVec(){}
		BitVec(int maxVal){
			hi = maxval;
			x = new int[1 + hi/BITSPREAD];
			for (int i=0; i<hi; i++)
				clr(i);
			n=0;
		}

	}
	
	int binarySearch(){
		// want: P(lo) is always false, and P(hi) always true
		int lo=0, hi=n-1, mi;
		while(lo+1<hi){
			mi= lo + (hi-lo)/2;
			if(P(mi)) 
				hi=mi; 
			else 
				lo=mi;
		}
		return hi;
	}

### Seg Tree

	typedef int TSegEl;
	struct SegmentTree {
		vector<TSegEl> A;
		vector<bool> dirty; //mark which nodes must be updated
		int size, n;
		vector<int> &data;
		SegmentTree(int n, vector<int> &array): n(n), data(array){
			size = 4*n+1;
			A.assign(size, -1);			
			dirty.assign(size, false);
			initialize(1, 0, n-1);
		}
		void initialize(int node, int start, int end){
			if (!dirty[node])
				return;
			if (start==end){
				A[node] = data[start];
			} else {
				int mid = (start+end)/2, 
						left = 2*node, 
						right = lef+1;
				initialize(left,start,mid);
				initialize(right,mid+1,end);
				A[node] = combine(A[left], A[right]);
			}		
			dirty[node] = false;
		}
		void initialize(){
			initialize(1, 0, n-1);
		}
		TSegEl combine(TSegEl valLeft, TSegEl valRight){
			if (valLeft==-1)
				return valRight;
			if (valRight==-1)
				return valLeft;
			if (data[valLeft] < data[valRight]){
			 return valLeft;
			} else {
				return valRight;
			}
		}
		TSegEl query(int node, int start, int end, int i, int j){
			if (i>end || j<start)
				return -1;

			if (start>=i && end<=j){ //interval contained
				if (dirty[node]) //if needs refresh
					initialize(node, start, end);
				return A[node];
			}

			int mid = (start+end)/2, 
					left = 2*node, 
					right = left+1;
			TSegEl valLeft = query(left,start,mid,i,j);
			TSegEl valRight = query(right,mid+1,end,i,j);
			return combine(valLeft, valRight);
		}

		void update(int node, int start, int end, int idx, int val){
			if (start > end || start>idx || end<idx) return;
			if (start==end){
				assert(start == idx);
				A[node] = val % P;
			} else {
				int mid = (start+end)/2, 
						left = 2*node, 
						right = 2*node+1;
				update(left,start,mid,idx, val);
				update(right,mid+1,end,idx, val);
				dirty[node] = true;
				//A[node] = combine(A[left], A[right], end-mid);
			}
			//printf("(%d, %d): %d -> %d\n", start, end, node, A[node]);
		}
		inline void update(int pos, int val){
			update(1, 0, n-1, pos, val);
		}
		inline TSegEl query(int i, int j){
			return this->query(1, 0, n-1, i, j);
		}
	};
	
### Integration

    int_a^b f(x) dx ~= (b-a)*f(a-b)/2 // rectangle rule
    int_a^b f(x) dx ~= (b-a)*(f(a)+f(b))/2 // trapezoidal rule
    int_a^b f(x) dx ~= (b-a)/6*(f(a)+4*f((a+b)/2)+f(b))/2 // simpson rule


## Coding style
	while while ((scanf scanf( %d , ("%d", n), n)  {  // read  input  + do value test together n), n) { … // read input + do value test together
	while (scanf("%d", n) != EOF) { … // read input and do EOF test

	isalpha (ctype.h)
	inline bool isletter(char c) { return (c>='A'&&c<='Z')||(c>='a'&&c<='z'); }
	abs (math.h)
	inline int abs(int a) { return a >= 0 ? a : -a; }
	pow (math.h)
	int int power( power(int int aa, int int b)b) {{
	int res=1; for (; b>=1; b--) res*=a; return res; }

## Graphs

### DFS

* O(V + E) if using Adjacency List
* O(V^2) if using Adjac
        
### BFS
	
	#define vi (vector<int>)
	#define tr(c,i) for(vi::iterator i = (c).begin(); i != (c).end(); i++)ency Matrix

	int dfs(vi& g, int a){
		int u, v;
		stack<int> s;
		vector<bool> vis(g.size(), false);
		s.push(a);
		vis[a] = true;
		
		while(!s.empty()){
			u = s.top(); s.pop();
			tr(g[u],it){
				if (!vis[ v = g[u][i] ]){
					vis[v] = true;
					vis.push(v);
				}
			}
		}
	
	}
	
	int bfs(vi& g, int a){
		int u, v;
		queue<int> q;
		vector<bool> vis(g.size(), false);
		s.push(a);
		vis[a] = true;
		
		while(!s.empty()){
			u = s.front(); s.pop();
			tr(g[u],it){
				if (!vis[ v = *it ]){
					vis[v] = true;
					vis.push(v);
				}
			}
		}
	
	}
	
### Connected Components

* DFS (and BFS) can find connected components: A call of dfs((u)) visits only vertices connected to u

### Topological sorting

#### Algo 1 (from start)

	L ← Empty list that will contain the sorted elements
	S ← Set of all nodes with no incoming edges
	while S is non-empty do
		remove a node n from S
		insert n into L
		for each node m with an edge e from n to m do
		    remove edge e from the graph
		    if m has no other incoming edges then
		        insert m into S
	if graph has edges then
		return error (graph has at least one cycle)
	else 
		return L (a topologically sorted order)
    

#### Algo 2 (from end)
    
	L ← Empty list that will contain the sorted nodes
	S ← Set of all nodes with no outgoing edges
	for each node n in S do
		visit(n) 
	function visit(node n)
		if n has not been visited yet then
		    mark n as visited
		    for each node m with an edge from m to n do
		        visit(m)
		    add n to L
        
### Eulerian Cycle

Walk on the graph edges of a graph which uses each graph edge in the original graph exactly once.

	Theorem: A directed graph possesses an Eulerian cycle iff
	1) It is connected
	2) For all {v} in {V} indegree(v) = outdegree(v)
	
#### Eulerian Path

A path from v1 to v2, that traverses every edge of G exactly once, although it may visit a vertex more than once.

	Theorem: A directed graph possesses an Eulerian path iff
	1) It is connected
	2) For all {v} in {V} indegree(v) = outdegree(v) with the possible exception of two vertices v1,v2 in which case,
		a) indegree(v1) = outdegree(v2) + 1
		b) indegree(v2) = outdegree(v1) - 1


### Hamiltonian Cycle

A Hamiltonian cycle is a graph cycle (i.e., closed loop) through a graph that visits each node exactly once.

### Kruskal


	struct Edge{
	  unsigned int u,v,w;
	  bool operator<(const Edge& e) const {
	  	return (w<e.w);
	  }
	  bool operator>(const Edge& e) const {
	  	return (w>e.w);
	  }
	};

	inline vector<Edge> kruskal(const vector<Edge>& edges, int V){
		vector<Edge> mst, nmst;
		priority_queue<Edge, vector<Edge>, greater<Edge> > q(edges.begin(), edges.end());
		unsigned int cicles[V];
		for(int i=0;i<V;i++) cicles[i] = i;

		while(mst.size()<(V-1) && q.size()){ //edges
			Edge s = q.top(); q.pop();
			if(cicles[s.u]!=cicles[s.v]){
				unsigned int number = cicles[s.v];
				for(int i=0;i<V;i++) {
		  			if(cicles[i]==number)
				    		cicles[i] = cicles[s.u];
				}
				mst.push_back(s);
			} else {
				nmst.push_back(s);
			}
		}
		while (q.size()){ //complementary spanning trees
			nmst.push_back(q.top()); 
			q.pop();
		}
		return mst;
	}

### Prim

* Greedy algorithm that finds a minimum spanning tree for a connected weighted undirected graph
* adjacency matrix, searching:	O(V2);
* binary heap and adjacency list:	O((V + E) log V) = O(E log V);
* Fibonacci heap and adjacency list:	O(E + V log V);

	inline int prim(){
		int ret = 0, v, w;
		priority_queue< pair<int, int>, vector< pair<int, int> >, greater< pair<int, int> > > q;
		vector<bool> vis(g.size(), false);

		q.push( make_pair(0,0) );
		while (!q.empty()){
			pair<int, int> p=q.top(); q.pop();
			v = p.second; //vertice
			w = p.first; //peso da aresta
			if (vis[v]) continue;
			vis[v] = true;
			ret += w;

			for (unsigned int i = g[v].size(); i--; )
			if ( !vis[ g[v][i].first ] ) {
				q.push(make_pair( g[v][i].second, g[v][i].first )); //peso, 
			}
		}
		return ret;
	}

### Dijkstra 

	vector<vector< pair<int, int> > > g; //(adjacent node, weight)
	#define edge(i, j, w) (g[i].pb(mp(j, w)); g[j].pb(mp(i, w));)

	int dijkstra(int a, int b){
		priority_queue< pair<int, int>, vector< pair<int, int> >, greater< pair<int, int> > > q;
		vector< int > d(g.size(), -1); //usa distancia!!!!!!!!

		pair<int, int> p; int v, w;
		q.push( make_pair(0,a) );
		while (!q.empty()){
			p=q.top(); q.pop();
			v = p.second; //vertice
			w = p.first; //peso da aresta
			if ( d[v] != -1 ) continue;
			if (v == b) return w;
			d[v] = w;

			for (unsigned int i = g[v].size(); i--; )
				if ( d[ g[v][i].first ] == -1 ) {
					q.push(make_pair( w+g[v][i].second, g[v][i].first )); //peso, 
				}
		}
		return -1;
	}
	
	typedef pair<int, int> PQData;
	vector< int > d; //distance
	vector< int > prev; //previous nodes

	inline int dijkstra(int a, int b){
		priority_queue< PQData, vector<PQData>, greater<PQData> > q;
		d.assign(g.size(), INF);
		prev.assign(g.size(), -1);
	
		PQData p;
		q.push( make_pair( d[a] = 0, a) );
		while (!q.empty()){
			int v = q.top().second;
		 	q.pop();

			for (int i = g[v].size(); i--; ){
				int alt = d[v]+g[v][i].second, u = g[v][i].first;
				if (d[u] > alt) {
					q.push(make_pair( d[u] = alt, u )); //new weight, vertex
					prev[u] = v;
				}
			}
		}
		if (d[b]!=INF)
			return d[b];
		return -1;
	}

	inline vector<int> getRoute(int dest){ //returns reverse path
		vector<int> path;
		path.push_back(dest);
		for (int u=dest; prev[u]!=-1; u=prev[u]){
			path.push_back(prev[u]);
		}
		return path;
	}
	

### Bellman Ford

	int bellman_ford(int a){
		vector< int > d(g.size(), INF);// Step 1: initialize graph
		vector< int > prior(g.size(), -1);
		int v, w, n = g.size();
		d[a] = 0;
		for (int i=0;i<n;i++){// Step 2: relax edges repeatedly
			for (int j=g[i].size(); j--;){
				v = g[i][j].first;
				w = g[i][j].second;
				if (d[v] > d[i] + w){
					d[v] = d[i] + w;
					prior[v] = i;
				}
			}
		}

		for (int v=g.size(); v--;){// Step 3: check for negative-weight cycles
			for (int j=g[v].size(); j--;){
				int u = g[v][j].first;
				if (d[u] > d[v] + g[v][j].second){ //g[i][j].second
					return -2; //"Graph contains a negative-weight cycle"
				}
			}
		}
		return 1;

	}
	
### Using strings as label for nodes

	vector<string> strings;
	map<string, int> stringsInt;

	int fetchString(string &s){
		return stringsInt[s];
	}

	int putString(string &s){
			if (stringsInt.find(s) == stringsInt.end()){
				int u = stringsInt[s] = strings.size();
				strings.push_back(s);
				return u;
			} else {
				return stringsInt[s];
			}
	}
	
### Floyd–Warshall O(V^3)

	int path[n][n]; //matrix initialized with edges weights
	for (int k=0; k<n; k++)
		for (int i=0; i<n; i++)
			for (int j=0; j<n; j++)
				path[i][j] = min(path[i][j], path[i][k]+path[k][j]);
				
	path[i][j] || (path[i][k] && path[k][j]); //transitive hull
	path[i][j] = min(path[i][j], max(path[i][k], path[k][j])); //minimax distance
	path[i][j] = max(path[i][j], min(path[i][k], path[k][j])); //maximin distance 
	path[i][j] = max(path[i][j], path[i][k] * path[k][j]); //safest path: path that maximizes the product of probabilities along the path


### A*

Used in pathfinding and graph traversal
TODO
...

## Dinamic Programming

### Knapsack

	for w from 0 to W do
	  T[0, w] := 0
	end for

	for i from 1 to n do
	  for j from 0 to W do
		if j >= w[i] then
		  T[i, j] := max(T[i-1, j], T[i-1, j-w[i]] + v[i])
		else
		  T[i, j] := T[i-1, j]
		end if
	  end for
	end for
	
### LIS (Longest Increasing Subsequence)

	int lisn2(vector<int> &v){
		vector<int> m(v.size(),0);
		int mx = 0;
		for (int i=0; i<v.size(); i++){
			for (int j=i; j--;)
				if (v[i]>v[j])
					m[i] = max(m[i], m[j]);
			m[i]++;
			mx = max(mx, m[i]);
		}
		return mx;

	}
	
	int lis(vector<int> &v){
		set<int> st;
		for (int i=0; i<v.size(); i++){
			st.insert(v[i]);
			set<int>::iterator it = st.find(v[i]);
			it++;
			if (it!=st.end())
				st.erase(it);
		}
		return st.size();
	}
	
### LCS
	
	int c[1001][1001], b[1001][1001];

	vector<int> order;
	void printLCS(int i, int j){
		if (!(i&&j)) return;
		if (b[i][j]==1){
			printLCS(i-1, j-1);
			order.push_back(i-1);
		} else if (b[i][j]==2){
			printLCS(i-1, j);
		} else
			printLCS(i, j-1);
	}

	int lcs(vector<string> &s1, vector<string> &s2){
		int m = s1.size(), n = s2.size();
	//	int c[m+1][n+1];
		for (int i=0; i<=m; i++)
			c[i][0] = 0;
		for (int j=0; j<=n; j++)
			c[0][j] = 0;
		for (int i=1; i<=m; i++)
			for (int j=1; j<=n; j++){
				c[i][j] = (s1[i-1] == s2[j-1])?
					c[i-1][j-1]+1
					:max(c[i][j-1], c[i-1][j]);
				b[i][j] = (s1[i-1] == s2[j-1])?1:
					(c[i][j-1]<=c[i-1][j]?2:3);//track previous word
			}
		
		printLCS(m, n);	//find the LCS
		int ok=false;
		for (int i=0; i<order.size(); i++){
			if (ok)
				cout << " ";
			else ok = true;
			cout << s1[order[i]];
		}
	
		return c[m][n]; //return the size of LCS
	}
		        
### Reduce Problem Set

	function LCS(X[1..m], Y[1..n])
		start := 1
		m_end := m
		n_end := n
		trim off the matching items at the beginning
		while start ≤ m_end and start ≤ n_end and X[start] = Y[start]
		    start := start + 1
		trim off the matching items at the end
		while start ≤ m_end and start ≤ n_end and X[m_end] = Y[n_end]
		    m_end := m_end - 1
		    n_end := n_end - 1
		C = array(start-1..m_end, start-1..n_end)
		only loop over the items that have changed
		for i := start..m_end
		    for j := start..n_end
		        the algorithm continues as before ...

### Reduce strings to hashes or Reduce the required space

### Counting change

	int coinsWays(int n){
		static int coins[] = {50,25,10,5,1}, ncoins = 5;
		vector<int> nway(n+1,0);
		nway[0] = 1;
		for (int i=0; i<ncoins; i++){
			int c = coins[i];
			for (int j=c; j<=n; j++)
				nway[j] += nway[j-c];
		}
		return nway[n];
	}


## Math

### Base Convertion

	long long convertFrom(string numb, int b){
		long long ret = 0, k=1;
		for (int i=numb.length(); i--;){
			int v;
			if (numb[i]>='0' && numb[i]<='9')
				v = numb[i] - '0';
			else
				v = numb[i] - 'A' + 10;
			if (v>=b) // invalid digit
				return -1;
			ret = ret + v*k;
			k *= b;
		}
		return ret;
	}

	string convertTo(long long numb, int b){
		string rt;
		if (!numb)
			return rt="0";
		int k=0;
		while (numb){
			int v = numb%b;
			if (v>9)
				rt+=(char)(v+'A'-10);
			else
				rt+=(char)(v+'0');
			numb/=b;
		}
		reverse(rt.begin(), rt.end());
		return rt;
	}

### GCD

	int gcd(int a, int b) {
		if (b == 0)
			return a;
		return gcd(b, a % b);
	}


### LCM

Least Common multiple

lcm(x,y) >= max(x,y)
lcm(x,y) <= x*y
lcm(x,y) = x*y/gcd(x,y)

### Exponentiation

	a^n = exp(log(a)*n) or pow(a,n)
	pow(a,n) = exp(log(a)*n)

### Modular arithmetics

	(x + y) mod n = ((x mod n) + (y mod n)) mod n
	x*y mod n = (x mod n) * (y mod n) mod n
	x^y mod n = (x mod n)^y mod n
	float(sqrt(n)) == ceil(sqrt(n))
	
	int myPow(int x, int p){
		if (p == 0) return 1;
		if (p == 1) return x;
		int tmp = myPow(x, p/2);
		if (p%2 == 0) return tmp * tmp;
		else return x * tmp * tmp;
	}
	
	
	
	long long powMod(long long x, long long p, int m){
		if (p == 0) return 1;
		if (p == 1) return x%m;
		long long tmp = powMod(x, p/2, m)%m;
		tmp = (tmp * tmp)%m;
		if (p%2 == 0) return tmp;
		else return (x * tmp)%m;
	}

### Stern-Brocot tree

TODO

### Recurrence

	a_n = a_{n-1} + 1, a_1 = 1 -> a_n = n
	a_n = 2*a_{n-1}, a_1 = 2 -> a_n = 2^n
	a_n = n*a_{n-1}, a_1 = 1 -> a_n = n!
	C_n = \sum^{n-1}_{k=0} C_k*C_{n-1-k}

### Combinatorics
	
	inline unsigned long long fact(int n){
		if (n==1 || n==0)
			return 1;
		int v=n;
		for (int i=n-1; i; i--)
			v*=i;
		return v;
	}
	
	//C(N, K)
	
	unsigned long long gcd(unsigned long long a, unsigned long long b) {
		if (b == 0)
			return a;
		return gcd(b, a % b);
	}

	void gcdDiv(unsigned long long& a,unsigned long long& b) { 
	  unsigned long long g=gcd(a,b); 
	  a/=g; 
	  b/=g; 
	} 

	unsigned long long C(int n,int k){ 
		unsigned long long numerator=1,denominator=1,toMul,toDiv,i; 
		if (k>n/2) k=n-k; /* use smaller k */ 
		for (i=k;i;i--) { 
			toMul=n-k+i; 
			toDiv=i; 
			gcdDiv(toMul,toDiv);       /* always divide before multiply */ 
			gcdDiv(numerator,toDiv);     
			gcdDiv(toMul,denominator); 
			numerator*=toMul; 
			denominator*=toDiv; 
		} 
		return numerator/denominator; 
	}

### Fibonacci

	if (n%2){
		v = n/2 + 1;
		unsigned long long int ta=fibo(v-1, mod)%mod, tb=fibo(v, mod)%mod, 
		ret = ((ta*ta%mod)+(tb*tb%mod))%mod;
		//fibcache[n] = ret; 
		return ret;
	} else {
		v = n/2;
		unsigned long long int ta=(2*fibo(v-1, mod))%mod, tb=fibo(v, mod)%mod, 
		ret = (((ta+tb)%mod)*tb)%mod;
		//fibcache[n] = ret; 
		return ret;
	}
	
	long int fib(unsigned long int n) {
       return lround((pow(0.5 + 0.5 * sqrt(5.0), n) - 
                      pow(0.5 - 0.5 * sqrt(5.0), n)) / 
                     sqrt(5.0));
    }
    
    long int fac(unsigned long int n) {
        return lround(exp(lgamma(n+1)));
    }
	

	
### Primes Numbers

Greatest prime that fits in a 32 bits int: (1<<31) - 1
	
Test only odd numbers up to sqrt(n)
	
	typedef unsigned long long TPrime;
	set<TPrime> primes;
	TPrime maxPrime = 0;
	int isPrime(TPrime num){
		if (num<=maxPrime)
			return primes.find(num) != primes.end();
		set<TPrime>::iterator it, itend = primes.upper_bound(sqrt(num));
		for (it=primes.begin(); it!=itend; it++){
			if (!(num % *it)) //found a divisor
				return false;
		}
		return true;
	}

	void genPrimes(TPrime mx){
		for (TPrime i=2; maxPrime<mx; i++){
			if (isPrime(i)){
				primes.insert(maxPrime = i);
			}
		}
	}

	map<TPrime, int> factors;
	void findFactors(TPrime n){ //trial divison method O(sqrt(n))
		factors.clear();
		for (set<TPrime>::iterator it=primes.begin(), ite=primes.upper_bound(sqrt(n)); it!=ite; it++){
			TPrime p = *it;
			if (p*p>n) break;
			while (!(n%p)){
				factors[p]++;
				n /= p;
			}
		}
		if (n>1){
			factors[n] = 1;
		}
	}
	
	//If a number N == a^i * b^j * ... * c^k then N has (i+1)*(j+1)*...*(k+1) divisors. 
	//Fermat algorithm: If 2^N modulo N = 2 then N has a high probability to be a prime number.

### Polar coordinate system

	x = r cos o
	y = r sen o
	r = sqrt(x^2 + y^2)
	o = arctan(y, x)
	
### LatlongDistance


	#define pi 3.14159265358979323846264
	#define toRad(x) ((x)*pi/180.0)

	double LatlongDistance(double lat1, double lon1, double lat2, double lon2){
		double R = 6371.0; // km
		double dLat = toRad(lat2-lat1);
		double dLon = toRad(lon2-lon1);
		lat1 = toRad(lat1);
		lat2 = toRad(lat2);

		double a = sin(dLat/2) * sin(dLat/2) +
	       		sin(dLon/2) * sin(dLon/2) * cos(lat1) * cos(lat2); 
		double c = 2 * atan2(sqrt(a), sqrt(1-a)); 
		return R * c;
	}
	
### Totient
	
	//	1, 1, 2, 2, 4, 2, 6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16, 6, 18, 8, 12, 10, 22, 8, 20, 12, 18, 12, 28
	//If n is a positive integer, then φ(n) is the number of integers k in the range 1 ≤ k ≤ n for which gcd(n, k) = 1.
	TPrime totient2(TPrime n){
		findFactors(n);
		TPrime tot = 1;
		for (map<TPrime, int>::iterator it=factors.begin(); it!=factors.end(); it++){
			tot *= (it->first-1)*myPow(it->first, it->second-1); //tot(n) = (p1-1).p1^(e1-1) . (p2-1).p2^(e2-1) ... (pr-1).pr^(er-1)  
		}
		return tot;
	}
	
### Carmichael Number
	//561,1105,1729,2465,2821,6601,8911,10585,15841,29341,41041,46657,52633,6274
5,63973
	//Carmichael number is a number which is not prime but has >= 3 prime factors

### Catalan Formula

	//1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900, 2674440
	//Cn = (2*n)!/(n+1)!n! = \prod_{k=2}^n (n+k)/k
	public static BigInteger catalan(int n){
	    
	    	BigInteger num = new BigInteger("1");
	    	BigInteger den = new BigInteger("1");
		for (int k=2; k<=n; k++){
			num = num.multiply(BigInteger.valueOf(n+k));
			den = den.multiply(BigInteger.valueOf(k));
		}
		return num.divide(den);
	}

## Geometry

	struct Point {
		T1 x;
		T1 y;
		Point(T1 x1, T1 y1): x(x1), y(y1) { }
		Point(){x = INF; y = INF;}
		bool operator <( const Point<T1>& p2 ) const{
			if (x<p2.x)
				return true;
			else if (x==p2.x)
				return y<p2.y;
			else
				return false;
		}
		inline double slope( const Point<T1>& p2 ) const{
			if ((x-p2.x) == 0)
				return INFINITY;
			return (y-p2.y)/(x-p2.x);
		}
		inline double distance(const Point<T1> &p1) const{
			return sqrt(pow(p1.x-x, 2) + pow(p1.y-y, 2));
		}
	};
	template<class T1>
	std::ostream& operator<<(std::ostream& stream, 
		             const Point<T1>& p) {
	   stream << "(" << p.x << "," << p.y << ")";
	    return stream;
	 }

	template<class T1>
	struct Line {
		T1 a; /* x-coefficient */
		T1 b; /* y-coefficient */
		T1 c; /* constant term */
		Line(const Point<T1> &p1, const Point<T1> &p2){ //two points to line
			if (p1.x == p2.x) {
				a = 1;
				b = 0;
				c = -p1.x;
			} else {
				b = 1;
				a = -(p1.y-p2.y)/(p1.x-p2.x);
				c = -(a*p1.x) - (b*p1.y);
			}
		}
		Line(const Point<T1> &p1, double m){ //point and slope to line
			a = -m;
			b = 1;
			c = -((a*p1.x) + (b*p.y));
		}
		Line (){}
		int insideLine(const Point<T1> &p){
			return 
				a*p.x +
				b*p.y +
				c
				== 0;
		}
		bool parallel(Line &l2){
			return (fabs(a-l2.a)<= EPSILON) && (fabs(b-l2.b)<= EPSILON);
		}
		bool same(Line &l2){
			return this->parallel(l2) && (fabs(c-l2.c) <= EPSILON);
		}
		T1 pointDistance(const Point<T1> &p){
			return fabs(
				a*p.x +
				b*p.y +
				c)/sqrt(a*a+b*b);
		}
		//Line intersection ????
		//Closest point2
		
	};
	
	template<class T>
	double signed_triangle_area(Point<T> a, Point<T> b, Point<T> c){
		return( (a.x*b.y - a.y*b.x + a.y*c.x - a.x*c.y + b.x*c.y - c.x*b.y) / 2.0 );
	}
	double triangle_area(Point<T> a, Point<T> b, Point<T> c){
		return( fabs(signed_triangle_area(a,b,c)) );
	}

	template<class T>
	bool collinear(Point<T> a, Point<T> b, Point<T> c) {
		return (fabs(signed_triangle_area(a,b,c)) <= EPSILON);
	}

	/*
	# Three points are a counter-clockwise turn if ccw > 0, clockwise if
	# ccw < 0, and collinear if ccw = 0 because ccw is a determinant that
	# gives the signed area of the triangle formed by p1, p2 and p3.*/
	inline double ccw(const Point<int> &p1, const Point<int> &p2, const Point<int> &p3){
	    return (double)(p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x);
	}
	    
	struct hull_comparator {
	  Point<int> r; //reference point
	  hull_comparator(const Point<int> &_r) : r(_r) {}
	  int operator()(const Point<int> &a, const Point<int> &b) {
		double v = ccw(r,a,b);
		if (v==0)
			if (r.distance(a) <= r.distance(b))
				return false;
			else
				return true;
		else if (v>0)
			return false;
		else
			return true;
	  }
	};
	    
	vector<Point<int> > hul;

	double grahamScan(vector<Point<int> > &points){
		int n = sz(points);
		//vector<Point<int> > hul;
		if (n<=1){
			return 0;
		} else if (n==2){
			return points.at(0).distance(points.at(1))*2;
		} else if (n==3){
			return points.at(0).distance(points.at(1))+points.at(1).distance(points.at(2))+points.at(0).distance(points.at(2));
		} else {
			hul.clear();
			sort(all(points));//WATCH(points);
			sort(points.begin()+1, points.end(), hull_comparator(points[0]));//WATCH(points);
			hul.pb(points[0]);
			hul.pb(points[1]);
			hul.pb(points[2]);
			for (int i=3; i<n; ++i){
				while (true){
					assert(sz(hul)>=2);
					if (ccw(*(hul.end()-2), *(hul.end()-1), points[i])>0)//positivo sai
						hul.pop_back();
					else {
						hul.pb(points[i]);
						break;
					}
				}
			}
			double sizeh = hul.begin()->distance(*(hul.end()-1));
			if (sz(hul)>2) tr1(hul, it){
				sizeh += it->distance(*(it+1));
			}
			return sizeh;
		}
	}
	
	#define DIMENSION 2 /* dimension of points */
	#define X		 0 /* x-coordinate index */
	#define Y	 	 1 /* y-coordinate index */
	typedef double point[DIMENSION];
	#define MAXPOLY		 200/* maximum number of points in a polygon */
	typedef struct {
		int n;		 	 /* number of points in polygon */
		point  p[MAXPOLY]; /* array of points in polygon */
	} polygon;
	typedef struct {
		point  p1,p2;		 /* endpoints of line segment */
	} segment;
	typedef point triangle[3]; /* triangle datatype */
	typedef struct {
		int n;		 	 /* number of triangles in triangulation */
		int t[MAXPOLY][3]; /* indicies of vertices in triangulation */
	} triangulation;
	
	bool point_in_box(point p, point b1, point b2) {
		return( (p[X] >= min(b1[X],b2[X])) && (p[X] <= max(b1[X],b2[X]))
		 	 &&  (p[Y]  >=  min(b1[Y],b2[Y]))  &&  (p[Y]  <=  max(b1[Y],b2[Y]))  );
	}
	bool segments_intersect(segment s1, segment s2) {
		line  l1,l2;	 	 /* lines containing the input segments */
		point  p;	 	 /* intersection point */
		   points_to_line(s1.p1,s1.p2,&l1);
		points_to_line(s2.p1,s2.p2,&l2);
		if (same_lineQ(l1,l2))  /* overlapping or disjoint segments */
			 	 return( point_in_box(s1.p1,s2.p1,s2.p2) ||
			 	 	 point_in_box(s1.p2,s2.p1,s2.p2)  ||
					point_in_box(s2.p1,s1.p1,s1.p2)  ||
					point_in_box(s2.p1,s1.p1,s1.p2)  );
		if (parallelQ(l1,l2)) return(FALSE);
		intersection_point(l1,l2,p);
		return( point_in_box(p,s1.p1,s1.p2) && point_in_box(p,s2.p1,s2.p2) );
	}

## printf format

	%[flags][width][.precision][length]specifier 

	specifier	Output	Example
	c	Character	a
	d or i	Signed decimal integer	392
	e	Scientific notation (mantissa/exponent) using e character	3.9265e+2
	E	Scientific notation (mantissa/exponent) using E character	3.9265E+2
	f	Decimal floating point	392.65
	g	Use the shorter of %e or %f	392.65
	G	Use the shorter of %E or %f	392.65
	o	Unsigned octal	610
	s	String of characters	sample
	u	Unsigned decimal integer	7235
	x	Unsigned hexadecimal integer	7fa
	X	Unsigned hexadecimal integer (capital letters)	7FA
	p	Pointer address	B800:0000
	n	Nothing printed. The argument must be a pointer to a signed int, where the number of characters written so far is stored.	
	%	A % followed by another % character will write % to stdout.	%

	The tag can also contain flags, width, .precision and length sub-specifiers, which are optional and follow these specifications:

	flags	descriptio
	
