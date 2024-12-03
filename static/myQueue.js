class Queue {
  constructor() {
    this.elements = new Array();
  }
  enqueue(element) {
    this.elements.push(element);
  }
  dequeue() {
    return this.elements.shift();
  }
  peek() {
    return this.elements[0];
  }
  get length() {
    return this.elements.length;
  }
  get isEmpty() {
    return this.length === 0;
  }
}
