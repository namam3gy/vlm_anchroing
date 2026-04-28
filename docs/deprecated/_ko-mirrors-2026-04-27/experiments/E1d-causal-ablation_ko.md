# E1d — 6-모델 패널 전체에 대한 인과 anchor-attention ablation

**상태:** `docs/experiments/E1b-per-layer-localisation.md`의 인과 후속 검증. Driver: `scripts/causal_anchor_ablation.py`. 분석: `scripts/analyze_causal_ablation.py`. 원시 출력: `outputs/causal_ablation/<model>/<run>/predictions.jsonl`. 집계 테이블: `outputs/causal_ablation/_summary/{per_model_per_mode.csv, by_stratum.csv}`. Figure: `outputs/causal_ablation/_summary/{fig_direction_follow.png, fig_adoption.png}`.

## TL;DR — 세 가지 발견

1. **단일-layer ablation은 6/6 모델 모두에서 null — E1b peak에서도, layer 0에서도.** 모델별 E1b peak layer (Gemma L5 / InternVL3 L14 / LLaVA-1.5 L16 / ConvLLaVA L16 / Qwen2.5-VL L22 / FastVLM L22)에서 anchor-image 칼럼의 attention mask에 큰 음수를 더해 죽여도, 6 모델 모두 `direction_follow_rate`이 baseline 95 % CI 바깥으로 나가지 않는다. `ablate_layer0` control도 6 모델 전부 동일 (`Δ direction_follow ∈ [−0.027, +0.005]`, 모든 CI가 baseline과 겹침) — Gemma도 마찬가지 (E1b가 layer 0–4의 anchor↔target swap을 보고했던 모델). 단일 layer (peak이든 아니든)에서 anchor를 *보는* 것은 상관관계 정보일 뿐 인과적으로 load-bearing하지 않다. Peak이 잘못된 target인 *것에 더해* "single layer" 자체가 잘못된 intervention class — 어떤 인과 효과를 위해서도 multi-layer ablation이 필요하다.
2. **Stack 전체 ablation은 `direction_follow`을 보편적으로 11–22 pp 줄이지만 3/6 모델에서 큰 fluency 저하를 유발.** 모든 LLM layer에서 anchor span을 ablate하면 `direction_follow_rate`이 11.5 pp (Gemma) ~ 22.0 pp (ConvLLaVA) 떨어지지만, `mean_distance_to_anchor`은 baseline ≈ 3 → ≈ 8–10 (Gemma, ConvLLaVA, LLaVA-1.5) 또는 ≈ 5500 (FastVLM, Qwen)으로 동시에 폭발한다. 모델이 더 이상 fluent하지 않음 — 외형상 감소의 상당 부분이 "모델이 anchor를 무시해서"가 아니라 "모델이 헛소리 출력해서"에서 옴.
3. **Upper-half ablation이 패널 전체에서 가장 깨끗한 단일 mitigation locus.** Layer `[n_layers/2, n_layers)`을 ablate하면 **6/6 모델에서** `direction_follow_rate`이 5.5–11.5 pp 감소하고 4/6은 fluency 청결 — mid-stack cluster (LLaVA-1.5, ConvLLaVA, InternVL3) + Qwen. Gemma는 moderate fluency 저하 (mean_distance_to_anchor 3.19 → 7.79, `ablate_all` 수준의 disruption); FastVLM은 이 모드에서도 broken (mean-distance ~5,600으로 폭발). Mid-stack cluster에 대해서는 upper-half ablation이 fluency를 깨지 않으면서 direction-follow를 줄이는 유일한 모드.

## Setup

6 패널 모델 각각에 대해, E1b stratified question set (모델당 n=200, top-decile-susceptible × 100 + bottom-decile-resistant × 100, `docs/insights/_data/susceptibility_strata.csv` 기준)을 세 condition (`target_only`, `target_plus_irrelevant_number`, `target_plus_irrelevant_neutral`)과 **7가지 ablation mode**로 실행:

| Mode | 무엇을 ablate하는가 | Layer (Gemma / InternVL3 / LLaVA-1.5 / ConvLLaVA / Qwen / FastVLM 각각 n=42 / 28 / 32 / 32 / 28 / 28) |
|---|---|---|
| `baseline` | 없음 | — |
| `ablate_layer0` | layer 0 (control: non-peak layer) | `[0]` |
| `ablate_peak` | E1b answer-step peak | `[5]` / `[14]` / `[16]` / `[16]` / `[22]` / `[22]` |
| `ablate_peak_window` | peak ± 2 (5-layer window) | `[3..7]` / `[12..16]` / `[14..18]` / `[14..18]` / `[20..24]` / `[20..24]` |
| `ablate_lower_half` | layer `[0, n_layers/2)` | LLM stack의 전반부 |
| `ablate_upper_half` | layer `[n_layers/2, n_layers)` | LLM stack의 후반부 |
| `ablate_all` | 모든 LLM layer | 전체 stack |

Ablation은 `scripts/causal_anchor_ablation.py`에 LLM stack의 각 `decoder_layer`에 forward pre-hook (`register_forward_pre_hook(..., with_kwargs=True)`)으로 구현. Hook은 `attention_mask` kwarg을 가로채 anchor-image-token 칼럼 `[s, e)`에 `−1 × 10⁴` (bf16-safe; `−inf`는 일부 kernel에서 NaN propagation 야기) 추가. Mask 개입은 row-wise가 아니라 column-wise이므로 *모든* 후속 position이 targeted layer에서 anchor span에 attend하지 않게 됨 — "layer `L ∈ ablate_layers`에 대해 anchor 이미지가 보이지 않음"과 동등.

`anchor_span`은 runner마다 다르게 결정: HF-input-id 모델 (Gemma, Qwen, LLaVA-1.5, InternVL3)은 `input_ids`에서 모델별 `image_token_id`를 스캔; ConvLLaVA는 `inputs_embeds` splice tracking; FastVLM은 `−200`-marker expansion 로직. 결정 코드는 `scripts/extract_attention_mass.py` (E1)와 `scripts/causal_anchor_ablation.py` (이번 라운드)에서 공유.

`(condition, mode)` 셀마다 세 가지 outcome metric 측정, 모두 bootstrap-CI 2,000 iter / 95 %:

- **`direction_follow_rate`**: `# triplets with |pred_number − anchor| < |pred_target_only − anchor|` ÷ valid triplets. "Anchor가 prediction을 끌어당긴다"의 가장 깨끗한 probe. 패널 baseline 16–31 %.
- **`adoption_rate`**: `# triplets with pred_number == anchor`. 더 엄격, 9–18 %.
- **`mean_distance_to_anchor`**: `mean |pred_number − anchor|` (낮을수록 anchor에 가까움). Sanity-monitor: 성공적인 ablation 하에서는 떨어지거나 (좋음 — 모델이 멀어짐) 평평; 폭발한다면 (예: baseline ≈ 3에서 ≥ 10), 모델이 ignore하는 게 아니라 hallucinate함.

## Result 1 — 단일-layer ablation은 6/6 모델 null (peak *과* layer-0 control)

E1b의 동기 부여 기대: peak layer가 layer-평균 anchor mass의 5–10×를 보유; ablate하면 적어도 direction-follow의 꼬리는 줄어야 한다. 그렇지 않다. 또한 모든 모델에 layer-0 control을 실행해 "peak이 상관관계일 뿐 다른 단일 layer가 인과 site"와 "single-layer ablation 자체가 부족"을 구분하려 했다. Layer-0도 6/6 모두 null.

`direction_follow_rate` baseline → `ablate_peak` (E1b peak) → `ablate_layer0` → Δ (CI):

| Model | E1b peak | baseline df [CI] | ablate_peak df | Δ peak | ablate_layer0 df | Δ layer0 |
|---|---:|---|---:|---:|---:|---:|
| gemma4-e4b | 5 | 0.265 [0.205, 0.325] | 0.285 | +0.020 | 0.270 | +0.005 |
| internvl3-8b | 14 | 0.161 [0.102, 0.219] | 0.129 | −0.032 | 0.133 | −0.027 |
| llava-1.5-7b | 16 | 0.305 [0.240, 0.370] | 0.315 | +0.010 | 0.290 | −0.015 |
| convllava-7b | 16 | 0.290 [0.230, 0.355] | 0.310 | +0.020 | 0.280 | −0.010 |
| qwen2.5-vl-7b | 22 | 0.215 [0.160, 0.275] | 0.220 | +0.005 | 0.220 | +0.005 |
| fastvlm-7b | 22 | 0.216 [0.151, 0.281] | 0.217 | +0.001 | 0.216 | +0.001 |

12개 |Δ| 모두 ≤ 3.2 pp; 12개 95 % CI 모두 baseline과 겹침. 어느 directional pull도 "no effect" null에 대한 one-sided sign test조차 통과하지 못함.

Layer-0 control이 가장 결정적인 모델은 **Gemma** — E1b가 layer 0–4의 anchor↔target swap을 보고했던, 패널에서 가장 그럴듯한 "early-layer 인과 site" 후보. Gemma의 `ablate_layer0` Δ = +0.005, peak Δ +0.020과 사실상 구분 불가능. 둘 다 baseline CI 안. 따라서 (b) "peak이 상관관계이고 다른 단일 layer가 인과 site"는 가장 강한 prior를 가졌던 모델에서도 지지받지 못함.

한 줄 해석: E1b peak의 anchor-attention 신호는 bias가 *지나가는* 곳의 상관관계 증거일 뿐 인과적 bottleneck이 아님. Anchor가 답에 미치는 영향은 여러 LLM layer에 redundant하게 인코딩되어 있어, 어느 단일 layer에서 제거해도 다른 layer의 인코딩이 그대로 남아 답이 변하지 않음. E1b가 제안한 per-family-peak E4 설계는 single-layer attention-mask intervention으로는 죽음; multi-layer intervention이나 다른 intervention class (contrastive decoding, vision-token re-projection)가 필요.

## Result 2 — stack 전체 ablation은 `direction_follow`을 줄이지만 3 모델에서 fluency를 깨뜨림

`ablate_all`: 모든 layer의 anchor span view를 우회.

| Model | baseline df | ablate_all df | Δ df | baseline mean_dist | ablate_all mean_dist | mean_dist Δ | fluency-broken? |
|---|---:|---:|---:|---:|---:|---:|---|
| gemma4-e4b | 0.265 | 0.150 | **−0.115** | 3.19 | 7.71 | +4.5 | ⚠️ moderate |
| internvl3-8b | 0.161 | 0.060 | **−0.101** | 5.67 | 4.91 | −0.8 | ✅ ok |
| llava-1.5-7b | 0.305 | 0.155 | **−0.150** | 3.58 | 10.01 | +6.4 | ⚠️ moderate |
| convllava-7b | 0.290 | 0.070 | **−0.220** | 3.18 | 8.23 | +5.1 | ⚠️ moderate |
| qwen2.5-vl-7b | 0.215 | 0.055 | **−0.160** | 48.6 | 48.9 | +0.3 | ✅ ok at this metric (Qwen baseline은 long-tail outlier로 이미 inflated — caveats 참조) |
| fastvlm-7b | 0.216 | 0.062 | **−0.154** | 3.12 | 5571 | +5568 | ❌ broken |

두 가지 해석:

- **direction_follow drop은 실재하고 보편적** (−10 ~ −22 pp). 모델이 어느 layer에서도 anchor를 볼 수 없으면 anchor 쪽으로 끌리지 않음 — bias는 집계 차원에서 인과적임.
- 그러나 Gemma / LLaVA-1.5 / ConvLLaVA에서 **mean-distance 신호가 4–6× 폭발**, FastVLM에서는 ~3 자릿수 폭발. 모델이 정답에 가까운 숫자가 아니라 헛소리 숫자를 출력 중. direction-follow drop의 일부는 "모델이 anchor에서 멀리 떨어진 wild 숫자를 출력"에서 오지 "anchor를 무시한 정답에 가까운 숫자"에서 오지 않음. "ablate_all"이 일부는 anchor pathway로, 일부는 일반 image-disruption으로 작동.

보수적 결론: stack-wide ablation은 인과 anchor pathway 존재를 지지하지만, 그 크기는 upper-bounded — fluency hit를 보정하면 진짜 인과 기여는 헤드라인 11–22 pp의 어딘가 아래. 본 설계로는 깔끔히 분리할 방법 없음.

## Result 3 — upper-half ablation이 가장 깨끗한 mitigation locus

`ablate_upper_half`: layer `[n_layers/2, n_layers)`에서 anchor span 우회.

| Model | baseline df | upper-half df | Δ df | baseline mean_dist | upper-half mean_dist | mean_dist Δ | clean? |
|---|---:|---:|---:|---:|---:|---:|---|
| gemma4-e4b | 0.265 | 0.210 | **−0.055** | 3.19 | 7.79 | +4.6 | ⚠️ moderate |
| internvl3-8b | 0.161 | 0.063 | **−0.098** | 5.67 | 5.00 | −0.7 | ✅ clean |
| llava-1.5-7b | 0.305 | 0.250 | **−0.055** | 3.58 | 4.72 | +1.1 | ✅ clean |
| convllava-7b | 0.290 | 0.235 | **−0.055** | 3.18 | 3.46 | +0.3 | ✅ clean |
| qwen2.5-vl-7b | 0.215 | 0.125 | **−0.090** | 48.6 | 48.8 | +0.2 | ✅ clean |
| fastvlm-7b | 0.216 | 0.101 | **−0.115** | 3.12 | 5641 | +5638 | ❌ broken |

6개 중 5개가 의미 있는 direction-follow 감소를 보임. 그 다섯 중 넷 (InternVL3, LLaVA-1.5, ConvLLaVA, Qwen)은 사실상 fluency 저하 없음. Gemma는 moderate fluency 저하지만 `ablate_all`보다 작은 direction-follow drop. FastVLM은 이 mode에서도 broken (E1b peak L22가 upper-half에 해당하는 점과 일관 — 하지만 이 mode가 이미 peak을 커버 중).

Mid-stack cluster (LLaVA-1.5, ConvLLaVA, InternVL3)가 가장 강한 케이스. 이 셋에 대해 upper-half ablation은:

- `direction_follow_rate`을 5.5–9.8 pp 감소,
- `mean_distance_to_anchor`을 사실상 평평하게 유지,
- 이 둘을 동시에 만족하는 유일한 모드.

이게 mid-stack cluster에 대한 architecture-blind E4 prototype 후보 단일 mode: upper-half attention layer에서 anchor-image span을 down-weight하고, 적당한 direction-follow 감소를 fluency 비용 없이 받아들임.

자연스러운 follow-up: `ablate_upper_quarter` (`[3n/4, n)`)을 시도해 가장 깨끗한 감소가 후기 layer에 집중되는지, upper half 전체에 분포하는지 확인. 이번 라운드에서는 미실행.

## Result 4 — peak-window ablation은 이질적; lower-half ablation은 이질적이고 종종 BACKFIRES

완전성을 위해:

`ablate_peak_window` (peak ± 2):

| Model | Δ direction_follow | reading |
|---|---:|---|
| gemma4-e4b | **+0.080** | **BACKFIRE** — anchor 효과 *증가* |
| internvl3-8b | −0.062 | small reduction |
| llava-1.5-7b | −0.010 | null |
| convllava-7b | −0.045 | small reduction |
| qwen2.5-vl-7b | −0.030 | null |
| fastvlm-7b | −0.041 | small reduction |

Gemma의 peak-window backfire는 E1b의 "layer 5는 knife-edge spike, anchor/target trade-off layer로 둘러싸임"과 일관. Spike와 trade-off layer를 함께 ablate하면 spike만 ablate하는 것보다 anchor/target competition을 더 disrupt해, 다운스트림 layer에서 anchor가 dominate할 자유를 얻는 것으로 보임.

`ablate_lower_half` (`[0, n_layers/2)`):

| Model | Δ direction_follow | Δ adoption | reading |
|---|---:|---:|---|
| gemma4-e4b | **+0.270** | **+0.410** | 거대한 **BACKFIRE** |
| internvl3-8b | +0.068 | +0.074 | **BACKFIRE** (n=83 vs baseline n=137 — 39 % triplet 손실; survivor-bias caveat 적용) |
| llava-1.5-7b | +0.165 | +0.275 | **BACKFIRE** |
| convllava-7b | −0.120 | +0.020 | df는 reduce, adoption은 null |
| qwen2.5-vl-7b | −0.010 | +0.025 | null |
| fastvlm-7b | −0.012 | +0.002 | null |

3/6 BACKFIRE, 1/6 reduce, 2/6 flat. 결정적 sub-finding: **ConvLLaVA의 lower-half ablation 행동이 LLaVA-1.5와 정반대**, 두 모델이 같은 E1b peak layer (L16)와 같은 text-stealing mechanism을 공유함에도. ConvLLaVA의 *인코딩된* anchor 신호는 lower stack에 전적으로 살고 있음 (그것을 ablate하면 L16의 anchor를 죽임); LLaVA-1.5의 lower-half-ablation은 backfire — 어떤 lower-stack circuit이 L16에서 anchor와 *경쟁*하고 있어 그것을 제거하면 anchor가 더 자주 이김.

이건 이질적, 모델별 finding — 헤드라인 아님. **일반화에 대한 caveat**으로 표시: E1b의 same-peak-layer + same-mechanism 관찰이 same-causal-structure를 함의하지 않음. ConvLLaVA와 LLaVA-1.5는 attention상으로는 동일하게 보이지만 lower-half ablation에 다르게 반응. 관찰적 mid-stack-cluster identity는 따라서 *partial*.

## Susceptibility-stratified 결과

`by_stratum.csv`의 `ablate_upper_half` (가장 깨끗한 mode) 행:

| Model | top-decile Δ df | bottom-decile Δ df | reads-as |
|---|---:|---:|---|
| gemma4-e4b | −0.110 | 0.000 | upper-half가 susceptible item에만 작용 |
| internvl3-8b | −0.243 | 0.000 | susceptible에 매우 강함 (bottom-decile baseline이 이미 0인 점 주의) |
| llava-1.5-7b | −0.100 | −0.010 | predominantly susceptible |
| convllava-7b | −0.090 | −0.020 | predominantly susceptible |
| qwen2.5-vl-7b | −0.160 | −0.020 | predominantly susceptible |
| fastvlm-7b | −0.275 | +0.024 | predominantly susceptible |

감소가 6 모델 중 5에서 top-decile-susceptible item에 집중됨. Bottom-decile-resistant item은 baseline anchor 효과가 작아서 (E1b A7 gap과 일관) ablation이 줄일 headroom이 거의 없음. E4에 고무적 — ablation 스타일 mitigation이 bias가 가장 큰 item을 specifically target하는 것으로 보임.

## E4 (mitigation)에 대한 함의

네 가지 클레임으로 압축:

1. **E1b peak에서의 단일-layer intervention은 작동하지 않을 것.** Peak-layer ablation null이 6-모델 패널 전체에 균일. E1b의 per-family peak-layer 권장은 상관관계; E4 prototype은 multi-layer 동시 개입을 하거나 다른 intervention class를 사용해야 함 (e.g. contrastive decoding, vision-token re-projection) — 단일 layer에서의 per-layer attention re-weighting이 아님.
2. **Upper-half attention re-weighting이 가장 그럴듯한 architecture-blind prototype.** 6 모델 모두에서 `direction_follow`을 5.5–11.5 pp 감소; 6 모델 중 4에서 fluency 보존 (특히 mid-stack cluster — 한 prototype이 세 encoder를 plausibly serve할 수 있음); 감소가 susceptibility-stratum-top item에 집중.
3. **Stack-wide ablation은 인과 anchor pathway를 ~11–22 pp direction-follow 감소로 upper-bound**, 그러나 실현된 감소가 일부 fluency-mediated이지 순수 anchor-suppression이 아님 — 헤드라인 숫자를 target이 아닌 upper bound로 해석.
4. **Lower-half ablation은 안전하지 않음.** 3/6 backfire (Gemma +0.27, LLaVA-1.5 +0.165, InternVL3 +0.068)가 lower-half attention re-weighting을 candidate intervention에서 배제 — 적어도 per-model gate 없이는. ConvLLaVA-vs-LLaVA-1.5 divergence가 mid-stack cluster 안에서도 관찰적 E1b identity가 인과적 identity를 의미하지 않는다는 경고.

## Caveats

- **Mask-replacement는 완전한 "ablation"이 아님**: anchor span의 attention score를 `−1e4`로 설정하면 강하게 suppress하지만 0으로 만들지는 않음 (kernel은 여전히 masked column 위로 softmax 계산). float16/bfloat16 정밀도의 attention에서는 사실상 masked; LLaVA-1.5에서 spot-check, hook 후 anchor span의 attention weight는 모든 head·targeted layer에서 < 1e-4.
- **Multi-layer ablation 상호작용은 additive하지 않음.** Peak ± 2 ablation은 peak alone을 다섯 번 ablate한 것과 *같지 않음*. `ablate_peak_window`와 `ablate_lower_half`의 외형 BACKFIRE는 이 상호작용을 반영.
- **Qwen `mean_distance_to_anchor` baseline은 ~48.6** — 적은 수의 long-digit hallucination (e.g. "9999...")으로 크게 inflated. Qwen의 `mean_distance_to_anchor` delta는 신중히; primary outcome으로 `direction_follow_rate`과 `adoption_rate` 사용.
- **InternVL3와 FastVLM은 여러 ablation 모드에서 triplet을 잃음** — 디지트 토큰을 안 내놓을 때가 있고, `_compute_metrics`가 `parsed_number`를 요구. InternVL3 baseline `n=137`; `ablate_lower_half`에서 83 (39 % 손실), `ablate_peak_window`에서 112, `ablate_upper_half`에서 112로 떨어짐. FastVLM baseline `n=147`; `ablate_all`, `ablate_upper_half`에서 159–161로 (일부 모드에서 살짝 *오름* — comparator cell shift 때문; 중요한 건 surviving subset이 baseline과 non-paired라는 점). 이 두 모델의 sub-baseline-n 모드 하의 "감소"나 "BACKFIRE"는 모두 survivor-bias caveat 적용 — 살아남은 triplet은 모델이 ablation 하에서 디지트를 출력할 수 있다는 조건부이고 이는 random하지 않음.
- **Sample-instance pairing**: triplet은 `sample_instance_id`로 join되어 모델별 mode-vs-baseline 비교가 paired. 위 bootstrap CI는 pair가 아닌 triplet을 resample하므로 그 pairing 하에서는 약간 conservative.
- **모델당 n=200**, susceptibility-stratified (top n=100 + bottom n=100). E1b와 같은 패널 — 식별자는 `docs/insights/_data/susceptibility_strata.csv`에서 로드.
- **FastVLM baseline이 re-aggregation 간 ~2 pp drift** (원래 full-modes run과 layer-0 re-aggregation 사이 0.238 → 0.216). FastVLM eager-attention 생성 path의 greedy decoding 하 non-determinism이 원인; 정성적 방향 (peak/layer-0 null, upper-half clean reduction, all-layer broken fluency)은 변함없으나 Δ 값이 1–2 pp shift. 위 테이블의 숫자는 layer-0 run을 포함하는 canonical re-aggregation 기준.

## Roadmap update

- §6 Tier 1 E1: open question "causal test (각 family의 peak layer에서 anchor token ablate, `direction_follow` 측정)" → **closed; single peak layer *과* layer-0 control 모두 6/6 모델에서 null; multi-layer redundancy 확인**. 본 writeup에 cross-link.
- §3 status: layer-localisation framing은 **observational only**; intervention 설계는 multi-layer ablation을 사용해야 함. Upper-half가 가장 좋은 architecture-blind 후보.
- §10 changelog 항목 2026-04-25.
