void string_to_string(int x) { cerr << x; }
void string_to_string(long x) { cerr << x; }
void string_to_string(long long x) { cerr << x; }
void string_to_string(unsigned x) { cerr << x; }
void string_to_string(unsigned long x) { cerr << x; }
void string_to_string(unsigned long long x) { cerr << x; }
void string_to_string(float x) { cerr << x; }
void string_to_string(double x) { cerr << x; }
void string_to_string(long double x) { cerr << x; }
void string_to_string(char x) { cerr << '\'' << x << '\''; }
void string_to_string(const char *x) { cerr << '\"' << x << '\"'; }
void string_to_string(const string &x) { cerr << '\"' << x << '\"'; }
void string_to_string(bool x) { cerr << (x ? "true" : "false"); }

template<typename T, typename V>
void string_to_string(const pair<T, V> &x) {
  cerr << '{'; 
  string_to_string(x.first); 
  cerr << ','; string_to_string(x.second); cerr << '}';
}
template<typename T>
void string_to_string(const T &x) {
  int foo = 0; 
  cerr << '{'; 
  for (auto &i: x) cerr << (foo++ ? "," : ""), string_to_string(i); 
  cerr << "}";
}

void debug_out() { cerr << "]\n"; }

template <typename T, typename... V>
void debug_out(T t, V... v) {
  string_to_string(t); 
  if (sizeof...(v)) cerr << ", "; 
  debug_out(v...);
}

#ifndef LOCAL
#define debug(x...) cerr << "[" << #x << "] = ["; debug_out(x)
#else
#define debug(x...)
#endif