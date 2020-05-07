class node {
 public:
  int id;
  node* l;
  node* r;
  node* p;
  bool rev;
  int sz;
  long long val;
  bool put;
 
  node(int _id) {
    id = _id;
    l = r = p = nullptr;
    rev = false;
    sz = 1;
    val = 0;
    put = false;
  }
 
  void unsafe_reverse() {
    rev ^= 1;
    swap(l, r);
    pull();
  }
 
  void unsafe_Apply(long long what) {
    val = what;
    put = true;
  }
 
  void push() {
    if (rev) {
      if (l != nullptr) {
        l->unsafe_reverse();
      }
      if (r != nullptr) {
        r->unsafe_reverse();
      }
      rev = 0;
    }
    if (put) {
      if (l != nullptr) {
        l->unsafe_Apply(val);
      }
      if (r != nullptr) {
        r->unsafe_Apply(val);
      }
      put = false;
    }
  }
 
  void pull() {
    sz = 1;
    if (l != nullptr) {
      l->p = this;
      sz += l->sz;
    }
    if (r != nullptr) {
      r->p = this;
      sz += r->sz;
    }
  }
};