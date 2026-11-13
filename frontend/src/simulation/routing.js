// Dijkstra: serbest akış kenar seyahat süresine (uzunluk / v0) göre en kısa süreli rota.
// Eşit maliyette sonuç deterministik kalsın diye düğüm indeksi ikinci sıralama ölçütüdür.

class MinHeap {
  constructor() { this.cost = []; this.node = []; }
  get size() { return this.node.length; }
  push(cost, node) {
    let i = this.node.length;
    this.cost.push(cost); this.node.push(node);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this._less(p, i)) break;
      this._swap(p, i); i = p;
    }
  }
  pop() {
    const topCost = this.cost[0], topNode = this.node[0];
    const lastCost = this.cost.pop(), lastNode = this.node.pop();
    if (this.node.length) {
      this.cost[0] = lastCost; this.node[0] = lastNode;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < this.node.length && !this._less(m, l)) m = l;
        if (r < this.node.length && !this._less(m, r)) m = r;
        if (m === i) break;
        this._swap(i, m); i = m;
      }
    }
    return [topCost, topNode];
  }
  _less(a, b) { return this.cost[a] < this.cost[b] || (this.cost[a] === this.cost[b] && this.node[a] <= this.node[b]); }
  _swap(a, b) {
    [this.cost[a], this.cost[b]] = [this.cost[b], this.cost[a]];
    [this.node[a], this.node[b]] = [this.node[b], this.node[a]];
  }
}

// Kaynaktan hedefe kenar indeks listesi ve serbest akış süresi; ulaşılamıyorsa null.
export function shortestRoute(graph, source, target) {
  if (source === target) return { edges: [], time: 0 };
  const n = graph.nodes.length;
  const dist = new Float64Array(n).fill(Infinity), prevEdge = new Int32Array(n).fill(-1);
  const heap = new MinHeap();
  dist[source] = 0; heap.push(0, source);
  while (heap.size) {
    const [d, u] = heap.pop();
    if (d > dist[u]) continue;
    if (u === target) break;
    for (const ei of graph.out[u]) {
      const e = graph.edges[ei], nd = d + e.freeFlowTime;
      if (nd < dist[e.to]) { dist[e.to] = nd; prevEdge[e.to] = ei; heap.push(nd, e.to); }
    }
  }
  if (!Number.isFinite(dist[target])) return null;
  const edges = [];
  for (let v = target; v !== source;) { const ei = prevEdge[v]; edges.push(ei); v = graph.edges[ei].from; }
  edges.reverse();
  return { edges, time: dist[target] };
}
