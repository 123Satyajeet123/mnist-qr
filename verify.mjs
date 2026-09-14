// Gate: the JS the QR carries must agree with the numpy reference exactly,
// on the same packed bytes. Run: node verify.mjs
import { readFileSync } from "node:fs";

const { blob, labels, ref, px } = JSON.parse(readFileSync("testset.json", "utf8"));
const { U, P } = new Function("atob", readFileSync("infer.js", "utf8") + ";return{U,P}")(
  s => Buffer.from(s, "base64").toString("binary"));

const bytes = Buffer.from(px, "base64");
const model = U(blob);
let agree = 0, correct = 0;
const wrong = [];
for (let i = 0; i < labels.length; i++) {
  const image = new Float32Array(784);
  for (let j = 0; j < 784; j++) image[j] = bytes[i * 784 + j] / 255;
  const got = P(image, model);
  if (got === ref[i]) agree++; else wrong.push(`#${i} js=${got} numpy=${ref[i]}`);
  if (got === labels[i]) correct++;
}
const n = labels.length;
console.log(`js/numpy agreement  ${agree}/${n}`);
console.log(`js accuracy         ${(correct / n * 100).toFixed(2)}%`);
if (wrong.length) { console.log("MISMATCHES:", wrong.slice(0, 10).join("  ")); process.exit(1); }
console.log("PASS");
