const { collide } = require("../pkg/collision");
const { performance, PerformanceObserver } = require("perf_hooks");

const size = 10_000;
const scale = 0.027;

const obs = new PerformanceObserver((list, obs) => {
  const entries = list.getEntries();
  console.log("Runs:", entries.length);
  console.log("Size:", size);
  console.log(
    "Duration:",
    list
      .getEntries()
      .map(({ duration }) => duration)
      .reduce((acc, val) => acc + val, 0.0) / entries.length,
  );
  obs.disconnect();
});

obs.observe({ entryTypes: ["measure"], buffered: true });

let points = new Float32Array(size);
for (let idx = 0; idx < size; idx += 4) {
  points[idx] = Math.random();
  points[idx + 1] = Math.random();
  points[idx + 2] = Math.random();
  points[idx + 3] = Math.random() * scale;
}

for (let i = 0; i < 5; i++) {
  performance.mark("start");
  collide(points);
  performance.mark("end");
  performance.measure("collide", "start", "end");
}
