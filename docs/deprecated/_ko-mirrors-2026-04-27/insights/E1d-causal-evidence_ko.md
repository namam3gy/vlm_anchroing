# E1d — peak-layer attention은 상관관계일 뿐, anchor의 인과 경로는 multi-layer다

**상태:** 6-모델 패널에서 E1b 후속 인과 검증. 원시 데이터: `outputs/causal_ablation/<model>/<run>/predictions.jsonl`. 집계 테이블: `outputs/causal_ablation/_summary/{per_model_per_mode.csv, by_stratum.csv}`. 전체 writeup: `docs/experiments/E1d-causal-ablation.md`.

## 클레임과 테스트

E1b는 anchor attention이 encoder family마다 단일 LLM layer에 집중된다고 보고했다 — Gemma L5, mid-stack cluster L14–16, Qwen·FastVLM L22. 자연스러운 후속: **그 layer에서 anchor에 attend할 능력을 제거**하면 모델의 anchor-pull 행동이 줄어드는가?

각 6 모델에 대해 6개의 layer set을 ablate한다 (single peak, peak ± 2, lower half, upper half, all layers, layer 0 as control). n=200 stratified, 3개 condition, `direction_follow_rate`을 baseline과 비교. 코드: `scripts/causal_anchor_ablation.py`, `scripts/analyze_causal_ablation.py`.

## 결과

| Mode | 무엇을 테스트하는가 | 결과 |
|---|---|---|
| `ablate_peak` | E1b의 헤드라인 | **6/6 모델 null** (\|Δ df\| ≤ 3.2 pp, 모든 CI가 baseline과 겹침) |
| `ablate_layer0` | non-peak control | **6/6 모델 null** (Δ df ∈ [−0.027, +0.005], 모든 CI가 baseline과 겹침) — Gemma 포함 (E1b가 L0–4의 anchor↔target swap을 보고했던 모델) |
| `ablate_all` | 인과 효과의 상한 | direction_follow −10 ~ −22 pp, 그러나 **3/6 모델에서 fluency 깨짐** (mean-distance 4–6× 또는 1000× 폭발) |
| `ablate_upper_half` | mitigation 후보 | **6/6 모델에서** −5.5 ~ −11.5 pp, 4/6은 fluency 청결 (mid-stack cluster + Qwen); Gemma는 marginal; FastVLM은 깨짐 |
| `ablate_lower_half` | 진단 | **이질적: 3/6 BACKFIRE, 1/6 reduce, 2/6 flat** |

가장 놀라운 결과는 **단일-layer ablation이 E1b peak에서도, layer 0에서도 null**이라는 점. E1b 단일-layer peak (layer-평균 anchor mass의 5–10×)이 명백한 E4 개입 site였고, Gemma의 L0–4 anchor↔target swap이 가장 그럴듯한 "early-layer 대안" 후보였다. 둘 다 단독으로는 인과적으로 load-bearing하지 않다.

## 해석

**(A) Multi-layer redundancy** 해석이 layer-0 control에 의해 확인됨. Anchor가 답에 미치는 영향은 LLM stack에 *redundant하게 인코딩됨* — 어느 단일 site에서 anchor view를 제거해도 (peak이든, layer 0이든, 그 사이 어디든, 6 모델 어느 모델이든) 나머지 stack이 그 인코딩을 복원함. Peak layer는 신호가 가장 잘 *보이는* 곳일 뿐, *고유하게 생산되는* 곳이 아니다. **(B) "peak이 상관관계일 뿐 다른 단일 layer가 인과 site"** 해석은 Gemma — E1b L0–4 anchor↔target swap으로 가장 그럴듯한 "early-layer 대안" 후보였던 — 에서도 지지받지 못함 (Gemma layer-0 Δ = +0.005, peak Δ = +0.020과 사실상 동일).

실용적 양성 결과: **`ablate_upper_half`만이 6-모델 패널 전체에서 작동**하고, mid-stack cluster의 fluency를 깨지 않는 단일 mitigation locus다. LLaVA-1.5, ConvLLaVA, InternVL3에서 upper-half ablation은 `direction_follow_rate`을 5.5–9.9 pp 줄이면서 `mean_distance_to_anchor`을 거의 변하지 않게 유지함. 이게 패널이 산출한 가장 강한 "architecture-blind" 후보 — 같은 depth에서 세 encoder (CLIP-ViT, InternViT, ConvNeXt)가 비슷하게 반응함.

## 왜 중요한가

실험 계획에서 두 가지가 바뀌고 한 가지는 유지된다.

**E4 설계의 변화.** E1b가 계획했던 per-family-peak attention re-weighting prototype은 폐기 — 단일-layer null이 peak에서도 layer 0에서도 6 모델 모두 성립함. 패널 어떤 architecture에서도 multi-layer 개입이 필요하다. 패널에서 작동하는 가장 단순한 E4 prototype은 *mid-stack cluster에 대한 upper-half anchor attention re-weighting*.

**페이퍼 클레임의 변화.** E1b/E1c의 "anchor가 LLM stack 어디에서 읽히는가" framing은 *상관관계 측면에서* 날카롭지만, *인과적인* per-family 개입 site를 시사하지 않는다. 페이퍼 텍스트는 다음을 구분해야:

- E1b finding: per-family peak layer는 attention 공간에서 anchor 신호가 집중되는 곳 (상관관계).
- E1d finding: 그 단일 layer는 단독으로는 인과적으로 load-bearing하지 않음 (single-layer attention-mask ablation null). Layer-0 control이 "E1b peak은 잘못된 단일 layer일 수 있다"는 대안을 배제 — single-layer ablation은 어느 layer에서든 부족함.
- E1d finding: upper-half가 패널에서 작동하는 단일 architecture-blind mitigation locus — mid-stack cluster + Qwen은 fluency 청결, Gemma는 marginal, FastVLM은 broken.

**유지되는 것.** E1b의 four-archetype encoder-family taxonomy는 여전히 *attention* signature의 가장 깨끗한 기술. 철회하지 않음. 단, 그것은 상관관계 정보일 뿐 single-layer causal intervention의 레시피는 아니라는 것.

## 단일 architecture cluster로 일반화하지 말라는 sub-finding

ConvLLaVA와 LLaVA-1.5는 E1b answer-step peak (L16), text-stealing budget source, 거의 동일한 magnitude를 공유. **그러나 lower-half ablation에서는 *반대로* 반응함.** ConvLLaVA `delta_df = −0.120`, LLaVA-1.5 `delta_df = +0.165`. Same-attention-signature가 same-causal-structure를 의미하지 않음. "mid-stack cluster"를 *인과적으로* 동질한 그룹으로 다루지 말라는 caveat — *attention-signature* 그룹으로서는 여전히 유용함.

## 이 결과가 *말하지 않는* 것

- **Anchor가 인과적이지 않다는 것 아님.** 인과적임. `ablate_all`은 direction_follow를 보편적으로 11–22 pp 감소시킴. Anchor는 집계 수준에서 인과적으로 load-bearing — 다만 그 효과 전체를 한 layer에 핀하지 못할 뿐.
- **Peak layer가 쓸모없다는 것 아님.** 각 모델에서 anchor signature가 어디 있는지 가장 강한 상관관계 marker임. 단지 single-layer 개입 site가 아닐 뿐.
- **모든 multi-layer ablation이 동등한 것 아님.** 그렇지 않음. `ablate_lower_half`는 3/6 모델에서 BACKFIRE (Gemma +27 pp, LLaVA-1.5 +16.5 pp). Multi-layer ablation은 *올바른 layer*가 필요하지, 단순히 multiple layers가 필요한 게 아님.

## 테스트하지 *않은* 것

- **Per-head sparsity.** Peak layer의 단일 attention head가 layer-aggregate가 아닌 곳에서 load-bearing할 수도 있음. 본 테스트의 strict generalisation이며 깨끗한 follow-up.
- **Vision-token re-projection** (anchor의 embedding을 LLM 진입 전에 0으로). E1d는 attention mask만 조작; anchor는 여전히 residual stream을 차지함. "anchor 입력을 완전히 거부"하는 intervention class는 미테스트.
- **Combination ablations** (peak + 보완 layer). Single-layer null은 올바른 조합이 `ablate_all` 효과를 더 청결하게 회복할 수 있음을 시사.

## 실험 계획에 대한 함의

- **E4 intervention class — open question 1 closed.** Single-layer attention re-weighting (E1b peak이든 layer 0이든) 은 후보에서 제외. Upper-half re-weighting이 prototype.
- **E4 architecture coverage.** Mid-stack cluster (LLaVA-1.5, ConvLLaVA, InternVL3)가 가장 leverage 높은 prototype target — 세 encoder, 한 공유 upper-half-clean 반응.
- **새 open question: multi-layer combinatorial ablation.** (peak에서 anchor span ablate) + (보완 layer 한 곳에서 ablate)의 합집합이 `ablate_all` magnitude의 감소를 더 낮은 fluency 비용으로 회복하는가? 비용 저렴; E4에 직접적으로 정보 제공.
- **Roadmap §6 E1 행:** open question "causal test" → **closed (single layer null; layer-0 control이 6/6 모델에서 multi-layer redundancy 확인)**. 새 open question: head-level sparsity. 새 open question: multi-layer combinatorial ablation.

## Caveats

- Bootstrap CI는 pair가 아닌 triplet 단위 — 약간 conservative.
- **InternVL3와 FastVLM은 여러 ablation 모드에서 triplet을 잃음** (digit 토큰 미출력). InternVL3 lower-half BACKFIRE는 n=83 vs baseline n=137 기준 — survivor-bias caveat 적용. FastVLM은 `ablate_all`, `ablate_upper_half`에서 비슷하게 triplet 감소. 그 항목들은 paired가 아닌 suggestive로 읽으라.
- Mask는 `−inf`가 아닌 `−1e4` (bf16-safe). LLaVA-1.5에서 spot-check: hook 후 anchor-column attention weight는 모든 head·targeted layer에서 < 1e-4. 개입은 경험적으로 강한 suppression이지 guaranteed zero는 아님.
- Susceptibility strata는 `docs/insights/_data/susceptibility_strata.csv`에서 로드; E1b와 동일.
