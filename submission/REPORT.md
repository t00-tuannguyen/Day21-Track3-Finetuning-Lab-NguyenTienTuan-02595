# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Tiến Tuân  **MSSV**: 2A202602595  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 (Colab Free, 14,6 GB khả dụng, fp16)

> Mọi con số trong report lấy từ `results/` (thí nghiệm chính, ViSFD) và
> `runs/default/results/` (thí nghiệm đối chứng trên corpus mặc định của lab). Con số nào
> **không** có trong file sẽ được ghi rõ nguồn ngay cạnh nó.

---

## 0. Lựa chọn thí nghiệm và lý do

| | Lựa chọn | Lý do |
|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` (mặc định tier T4) | Vừa T4 ở bf16/fp16 LoRA mà không phải dùng QLoRA (deck §13); giữ mặc định để so được với số đo tham chiếu của lab |
| Dataset chính | **UIT-ViSFD** — bình luận điện thoại thật → JSON cảm xúc theo khía cạnh (B2) | Dữ liệu thật, do người gán nhãn, văn nói/teen-code; không gian nhãn là quy ước riêng của bộ dữ liệu → đúng loại kiến thức fine-tune *nên* mang lại. Chi tiết, khử nhiễm: `data/CUSTOM_DATASET.md` |
| Dataset đối chứng | Corpus mặc định: 250 ticket CSKH → JSON triage | Chạy trước để quen pipeline, và để có một điểm so sánh: **cùng model, cùng cấu hình, chỉ khác dữ liệu** |

Hai bộ dữ liệu dùng chung tập `eval_regression.jsonl` (15 câu hỏi phổ thông, không đổi),
nên cột regression so sánh trực tiếp được giữa hai thí nghiệm.

**Thay đổi code để chạy ViSFD** (nhánh `visfd`): công tắc task (`data/TASK` / `LAB_TASK`);
prompt (a)/(b) riêng cho ViSFD; scorer `aspect_f1` (F1 trên cặp *(khía cạnh, cực)*) và
`aspect_format`; NB2 lưu thêm `baseline_preds.json` để NB5 đặt (b) cạnh fine-tune trong
bảng định tính. Ngưỡng của cổng hồi quy, tập regression và quy trình train **không đổi**.

---

## 1. Setup

| | ViSFD (chính) | Mặc định (đối chứng) |
|---|---|---|
| Dataset | 250 train / 50 eval từ UIT-ViSFD (`train` / `test` split gốc) | 250 ticket CSKH tổng hợp |
| Train / val | 225 / 25 (seed 42) | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được **128**, gợi ý 256 | 1024 — p95 đo được **98**, gợi ý 256 |
| `MASK_MODE` | `assistant-only` | `assistant-only` |
| Epochs / max_steps | 2 / **30** (batch hiệu dụng 1×16) | 2 / **30** |
| LoRA `correct` | text-linear · r=16 · α=32 · LR 1e-4 · fp16 | giống hệt |

**Vì sao giữ `max_length=1024` dù p95 gợi ý 256?** Batch per-device = 1 nên không có
padding: `max_length` chỉ là ngưỡng cắt. Mẫu dài nhất là 150 token (ViSFD) / 101 token
(mặc định), nên ngưỡng 1024 không bao giờ chạm tới và không tốn thêm tính toán. Hạ xuống
256 cho kết quả y hệt; tôi giữ giá trị của tier để không đổi thêm một biến so với cấu
hình tham chiếu.

**Template có giữ khối `<think>` không?** **Có** — `results/template_check.json`:
`"reasoning preserved — safe to train on traces"`. Dữ liệu của tôi không có reasoning
trace (đáp án là JSON trần), nên điều này không ảnh hưởng tới mask; nhưng nó giải thích
vì sao đoạn được tính loss bắt đầu bằng `</think>` (mục 2).

---

## 2. Mask proof (NB1)

| | ViSFD | Mặc định |
|---|---|---|
| `supervised_fraction` | **0.1966** (23 / 117 token) | 0.4149 (39 / 94 token) |
| Câu trả lời nằm trong loss | `true` | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` | `true` |

Đoạn được tính loss (ViSFD, `results/mask_proof.json → supervised_preview`):

```
</think>

{"features": "positive", "battery": "positive", "ser_acc": "positive"}<|im_end|>
```

Đoạn bị che gồm system prompt, toàn bộ bình luận của người dùng và `<think>\n\n` mà
template tự chèn. Model chỉ học **đóng khối think rỗng → viết JSON → `<|im_end|>`**.
Việc `</think>` nằm trong loss là đặc điểm của template Qwen3.5 (generation prompt mở
`<think>` rồi để assistant đóng) chứ không phải lỗi mask: phần câu hỏi hoàn toàn không
có trong loss, và `supervised_fraction` thấp hơn nhiều so với ngưỡng 0.95.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

**ViSFD** (`results/verdict.json → comparison`, n = 50):

| Run | target (F1) | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3212.4 |
| (b) base + optimized prompt | **0.687** | 0.7911 | **1.000** | **821.0** |
| (c) LoRA fine-tune | 0.6374 | 0.2800 | 0.9833 | 1061.7 |

**Mặc định** (`runs/default/results/verdict.json`, n = 50):

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3128.8 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 979.0 |
| (c) LoRA fine-tune | **0.970** | 0.6111 | 1.000 | 1384.1 |

**(b) có thật sự mạnh hơn (a) không?** **Có, rất rõ** ở cả hai bộ: target 0 → 0.687 /
0.765, format 0 → 1.0, và nhanh hơn ~4 lần. Với prompt naive, base model không trả JSON
mà viết văn xuôi dài — vừa sai định dạng vừa chậm.

**Tôi có sửa `OPTIMIZED_PROMPT` không?** Với corpus mặc định: **không** (`make verify`:
"baseline (b) prompt unmodified"). Với ViSFD: tôi viết **một prompt (b) mới** vì task khác
hẳn, **trước khi** đo bất kỳ baseline nào, theo hướng **làm mạnh**: định nghĩa từng khía
cạnh bằng ví dụ cụ thể, định nghĩa `neutral`, quy tắc "chỉ ghi khía cạnh được nhắc tới",
và 2 ví dụ few-shot tự viết (không lấy từ split nào). Bằng chứng nó là một mốc mạnh: chỉ
với prompt, base model đã đạt F1 0.687, và — tính từ `results/baseline_preds.json` so với
nhãn — phát hiện đúng khía cạnh với precision 0.85 / recall 0.82, và đúng cả cặp *(khía
cạnh, cực)* với precision 0.73 / recall 0.70: lỗi của (b) chia khá đều giữa sai khía cạnh
và sai cực.

---

## 4. Giải phẫu cấu hình sai (NB4, chấm ở NB5 §4)

**ViSFD** (`results/runs.csv` + `results/autopsy.json`, 30 step mỗi run):

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | train s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 1.2107 | **0.6374** | 0.983 | 413.8 | 8.97 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | **1.0753** | **0.6600** | 0.980 | 259.6 | 8.99 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 2.5101 | **0.0000** | 0.000 | 390.2 | 8.98 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 1.2308 | **0.5373** | 0.872 | 460.3 | **4.05** |

**Mặc định** (`runs/default/results/`):

| Run | train loss | **target** | format | train s | VRAM GB |
|---|---|---|---|---|---|
| `correct` | 0.6269 | **0.970** | 1.0 | 402.3 | 8.78 |
| `attn_only` (r=283) | **0.5380** | **0.970** | 1.0 | 258.5 | 8.79 |
| `wrong_lr` | 1.5702 | **0.000** | 0.0 | 384.8 | 8.78 |
| `qlora` | 0.7058 | **0.940** | 1.0 | 451.8 | 3.86 |

Mỗi run chỉ đổi **một** biến so với `correct` (rubric 2.3): `attn_only` đổi **vị trí**
(rank được nâng lên để giữ nguyên số tham số, lệch 0,025%); `wrong_lr` đổi **learning
rate** (1e-4 → 1e-5); `qlora` đổi **độ chính xác của base** (16-bit → 4-bit NF4). Cùng
mask, cùng dữ liệu, cùng 30 step — `make verify` kiểm tra cả hai điều kiện ngân sách.

**4.1 — `attn_only` có cùng số tham số với `correct`. Trên tập target nó thắng, thua hay
hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về rank so với vị trí?**

Trên target, `attn_only` **hoà** với `correct` ở corpus mặc định (0.970 = 0.970) và **nhỉnh
hơn** ở ViSFD (0.660 vs 0.637). Chênh 0.023 trên 50 mẫu nằm trong mức dao động giữa hai lần
chạy cùng cấu hình mà tôi quan sát được (mục 8), nên kết luận trung thực là **không đo được
khác biệt về vị trí** ở ngân sách tham số bằng nhau — tôi không tái lập được nhận định của
deck §11.2 trên model này. Thứ tự theo train loss lại *luôn* xếp `attn_only` lên đầu (1.075 <
1.211; 0.538 < 0.627). Ở corpus mặc định, nếu chấm bằng loss tôi sẽ kết luận "attention-only
tốt hơn rõ rệt", trong khi trên tác vụ hai run hoà nhau — đúng Lỗi #3: chỉ số thay thế phóng
đại một khác biệt không tồn tại. Về rank vs vị trí: `attn_only` dồn ngân sách vào r=283 trên
rất ít module (q,v chỉ có ở các lớp full-attention của kiến trúc lai Qwen3.5) mà không thua;
nghĩa là với 225 mẫu và 30 step, **cả rank lẫn vị trí đều không phải nút thắt** — ngân sách
32 triệu tham số đã thừa so với lượng thông tin trong dữ liệu. Lợi ích thực tế duy nhất đo
được của `attn_only` là chi phí: train nhanh hơn ~37% và suy luận nhanh hơn ~39%
(652.5 vs 1061.7 ms trên ViSFD).

**4.2 — `wrong_lr` chỉ khác một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà
không biết LR, bạn sẽ kết luận sai điều gì?**

Loss cuối của `wrong_lr` cao gấp đôi `correct` (2.510 vs 1.211 trên ViSFD; 1.570 vs 0.627 ở
mặc định), và trên tác vụ nó **sụp hoàn toàn**: target 0.000, format 0.000, latency 5205 ms
(dài gần 5 lần) — dấu hiệu model vẫn viết văn xuôi tới hết `max_new_tokens` như base model
với prompt naive (suy ra từ format 0 và latency; output của run đối chứng không được lưu).
Nói cách khác, 30 step ở LR thang full-FT gần như không dịch chuyển được hành vi. Nếu
chỉ thấy đường loss giảm chậm mà không biết LR, tôi sẽ kết luận sai rằng "dữ liệu quá khó" hoặc
"LoRA không đủ năng lực" và đi tăng rank hay tăng epoch — tức là sửa sai chỗ. Đây là nút có
hiệu ứng lớn nhất trong cả bốn run: đổi đúng một con số làm điểm rơi từ 0.64 về 0.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo có ủng hộ khuyến nghị "không
dùng QLoRA cho dòng model này" không?**

QLoRA giảm peak VRAM từ 8.97 xuống **4.05 GB (−55%)** trên ViSFD (8.78 → 3.86 GB ở mặc định).
Cái giá: target giảm **0.100** trên ViSFD (0.637 → 0.537) và 0.030 ở mặc định; format tụt còn
0.872 (model bắt đầu sinh JSON hỏng/khoá lạ); và nó **chậm hơn** chứ không nhanh hơn — train
460 s vs 414 s, suy luận 1789 vs 1062 ms vì phải giải lượng tử ở mỗi lớp. Đáng chú ý là train
loss gần như không đổi (1.231 vs 1.211) — loss **che mất** cái giá của lượng tử hoá, chỉ eval
trên tác vụ mới thấy. Số đo của tôi **ủng hộ** khuyến nghị của nhà cung cấp: trên T4, bản
16-bit đã vừa với dư ~5 GB, nên QLoRA chỉ mang lại thiệt hại; nó chỉ đáng cân nhắc khi model
không vừa GPU ở 16-bit.

**Phân tích vị trí vs rank — đâu là đòn bẩy (rubric 2.4).** Xếp theo độ lớn hiệu ứng trên
target ViSFD: learning rate (−0.637) ≫ lượng tử hoá (−0.100) ≫ vị trí ở ngân sách bằng nhau
(+0.023, trong nhiễu). Và cả bốn đều nhỏ hơn hiệu ứng của **dữ liệu**: cùng cấu hình `correct`
cho +0.205 trên corpus tổng hợp nhưng −0.050 trên ViSFD (mục 7).

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy (ViSFD)**: `FAILED`
`target Δ = −0.050` · `regression Δ = −0.511` · `valid_trace_rate = 0.0`

(Đối chứng mặc định: `FAILED`, `target Δ = +0.205`, `regression Δ = −0.180`.)

Bản fine-tune trên ViSFD trượt **cả hai** điều kiện của cổng. Về target, nó đạt F1 0.637,
thấp hơn baseline (b) 0.050: trên 50 mẫu eval, fine-tune thua (b) ở 23 mẫu, thắng ở 12, hoà
15 (`results/qualitative.json`). Về năng lực chung, điểm regression rơi từ 0.791 xuống 0.280 —
mất gần hai phần ba — xa ngưỡng cho phép 0.02. Định dạng gần như không phải vấn đề (0.983),
nên theo thứ tự chẩn đoán của NB5 lỗi không nằm ở template/mask; LR cũng đúng thang (run
`wrong_lr` cho thấy LR sai trông như thế nào, và đây không phải trường hợp đó). Còn lại hai
nguyên nhân: (1) **quên thảm hoạ** — model học rằng *mọi* đoạn văn tiếng Việt đều phải trả
về JSON khía cạnh; và (2) **prompt engineering đã thắng** — prompt (b) mang định nghĩa của
từng khía cạnh vào ngữ cảnh, còn bản fine-tune được eval với prompt naive nên phải tự rút ra
quy ước nhãn từ 225 ví dụ, trong đó có khía cạnh chỉ xuất hiện 3 lần (`storage`, xem
`data/visfd_build.json`). `valid_trace_rate = 0.0` là đúng kỳ vọng vì dữ liệu không có
reasoning trace. Kết luận: với bài toán này, ở quy mô dữ liệu này, **fine-tune không đáng
triển khai** — một prompt tốt cho kết quả tốt hơn, rẻ hơn (821 vs 1062 ms) và không phá huỷ
năng lực chung của model.

---

## 6. Định tính — có cả ca THUA

Lấy từ `results/qualitative.json` (đã xếp theo `ft_score − b_score`); bình luận rút gọn.

| # | Bình luận (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "Sao máy sạt khá nóng mặt dù kh bật mạng… mới có 48% thấy nóng rồi" | battery: neg | performance: neg → **0.00** | battery: neg, performance: neg → **0.67** | ✅ FT thắng — hiểu "sạt"/"48%" là pin theo quy ước ViSFD |
| 2 | "Rất ổn, để test thêm vài tuần xem sao…" | general: pos | general: **neutral** → 0.00 | features: pos, general: pos → **0.67** | ✅ FT thắng — (b) quá dè dặt, gán neutral cho lời khen |
| 3 | "Sản phẩm đẹp, rễ sử dụng, rất tốt, nhân viên giao hàng… nhanh và nhiệt tình" | general: pos, ser_acc: pos | design: pos, ser_acc: pos, general: neutral → 0.40 | features: pos, ser_acc: pos, general: pos → **0.80** | ✅ FT thắng |
| 4 | "Không còn gì tuyệt vời hơn khi pin iphone 7 tuột nhanh bá đạo… màn hình rổ, loa bật max máy rung" | screen: neg, features: neg, battery: neg, price: neutral | battery/screen/features: neg → **0.86** | battery: **pos**, price: pos, ser_acc: neg, **`ser_gen`**: neg → **0.00** | ❌ **FT thua** — không nhận ra mỉa mai, và bịa ra khoá `ser_gen` không có trong schema |
| 5 | "Hơn chục triệu mà mặt lưng nhựa, chả bằng con Bphone3 7 triệu…" | design: neg, price: neutral | design: neg → **0.67** | screen: neutral, price: neg, general: neg → **0.00** | ❌ **FT thua** — thêm khía cạnh không được nhắc, bỏ sót `design` |
| 6 | "Sau hơn 3 tuần sử dụng Xiaomi note 9s thì tôi không thể gửi được tin nhắn…" | features: neg, ser_acc: neutral | ser_acc: neutral → **0.67** | camera, battery, performance: neutral, general: neg… → **0.00** | ❌ **FT thua** — "phun" ra cả loạt khía cạnh không liên quan |

**Có mẫu chung nào ở các ca FT thua không?** Có, hai mẫu rõ ràng. (1) **Dự đoán thừa khía
cạnh**: ở cả ba ca thua, fine-tune liệt kê những khía cạnh bình luận không hề nhắc tới
(camera, battery, performance cho một lời phàn nàn về tin nhắn). Prompt (b) có quy tắc
tường minh "chỉ ghi khía cạnh có nhắc tới"; bản fine-tune phải tự suy ra quy tắc đó từ dữ
liệu và chưa học được. Ngay cả ở ca thắng (#1, #2) nó vẫn thêm một khía cạnh thừa, nên không
đạt điểm tối đa. (2) **Không xử lý được câu mỉa mai và câu so sánh** (#4, #5) — base model
với prompt tốt giữ được khả năng đọc hiểu ngữ dụng này, còn fine-tune 30 step đã "đè" nó bằng
thói quen ánh xạ từ khoá → cực. Ca #4 còn cho thấy fine-tune vi phạm schema (`ser_gen`) —
một ví dụ của phần format bị mất (0.983 thay vì 1.000).

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không** nên deploy bản fine-tune ViSFD này. Nó thua prompt (b) trên
chính tác vụ được huấn luyện (F1 0.637 vs 0.687), chậm hơn (1062 vs 821 ms) và làm hỏng năng
lực chung của model (regression 0.791 → 0.280). Điều thú vị là cùng một cấu hình, trên
corpus tổng hợp của lab, lại *thắng* (b) rất rõ (0.970 vs 0.765). Sự khác biệt đó cho tôi
biết đòn bẩy thật sự nằm ở đâu. Corpus mặc định được sinh từ một bộ mẫu câu cố định, nên
train và eval đến từ cùng một "máy sinh"; 30 step LoRA đủ để model học thuộc máy sinh đó.
ViSFD là văn nói thật, nhãn mang quy ước của người gán, có khía cạnh hiếm và câu mỉa mai;
225 ví dụ không đủ để model tự rút ra những quy tắc mà prompt (b) có thể *nói thẳng* ra. Vì
vậy, xếp theo độ lớn hiệu ứng tôi đo được: **dữ liệu** (đổi corpus làm Δ target đảo dấu từ
+0.205 sang −0.050) và **learning rate** (sai một bậc → target về 0) là hai đòn bẩy thật;
lượng tử hoá là một cái giá vừa phải (−0.10); còn **vị trí adapter ở ngân sách bằng nhau
không tạo ra khác biệt đo được**. Mask thì đúng ngay từ đầu nên không phải yếu tố ở đây —
nhưng chính vì đã *chứng minh* mask đúng ở NB1 mà tôi có thể loại nó khỏi danh sách nghi
phạm khi phân tích thất bại. Cuối cùng, regression sụp nặng hơn hẳn trên ViSFD (−0.511 so với
−0.180). Giả thuyết của tôi — chưa kiểm chứng — là ticket tổng hợp có hình thức rất dễ nhận
ra ("Shop ơi… mã đơn…"), nên model còn phân biệt được lúc nào *không* phải trả JSON; còn bình
luận ViSFD trông giống mọi đoạn tiếng Việt khác, nên model tổng quát hoá thành "gặp tiếng
Việt là trả JSON". Nếu đúng, replay 1–5% dữ liệu phổ thông (deck §6.3) là cách sửa đầu tiên
cần thử.

<!-- NHÁP do AI viết dựa trên quá trình làm lab — Tuân hãy sửa lại bằng trải nghiệm và lời của mình (rubric 4.4 chấm độ cá nhân, cụ thể). -->

**Ba điều tôi học được:**
1. **Train loss có thể nói dối theo cả hai chiều.** `attn_only` có loss thấp nhất ở cả hai
   corpus nhưng chỉ hoà trên tác vụ; `qlora` có loss gần bằng `correct` nhưng mất 0.10 F1. Nếu
   tôi dừng ở bảng loss của NB4, tôi đã rút ra hai kết luận sai trong cùng một bảng.
2. **Cùng một cấu hình "đúng" có thể thắng hoặc thua tuỳ dữ liệu.** Tôi đã định báo cáo
   "fine-tune thắng prompt +0.205" sau lần chạy đầu. Chỉ khi đổi sang dữ liệu thật tôi mới
   thấy con số đó phần lớn đến từ việc corpus tổng hợp quá dễ học thuộc, không phải từ sức
   mạnh của LoRA.
3. **Một prompt (b) được viết tử tế là một đối thủ rất khó.** Prompt của tôi cho F1 0.687,
   format 1.0, nhanh gấp 4 lần prompt naive — mà không tốn một step huấn luyện nào. Đo nó
   *trước* khi train là điều duy nhất ngăn tôi vô thức làm yếu nó đi sau khi thấy fine-tune thua.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** (1) trộn 3% câu hỏi phổ thông (replay) vào tập train
ViSFD để kiểm chứng giả thuyết về regression; (2) train với chính prompt (b) làm system prompt
(thay vì prompt naive) để model không phải tự suy ra định nghĩa khía cạnh — đo xem target có
vượt 0.687 không; (3) chạy lại `correct` và `attn_only` với 2–3 seed để biết chênh lệch 0.023
giữa chúng có thật hay chỉ là nhiễu.

---

## 8. Ghi chú về độ tin cậy

* **Dao động giữa các lần chạy.** Thí nghiệm ViSFD đã chạy trọn vẹn **hai lần** trên T4 với
  cùng code và seed. Lần đầu bị mất máy ảo trước khi tải được `results/`; các con số của lần
  đó tôi chỉ đọc được từ log trên màn hình (**không có file**): (c) target 0.651, regression
  0.267, `attn_only` 0.648. Lần thứ hai (file trong `results/`): 0.637 / 0.280 / 0.660. Baseline
  (a), (b) giống hệt nhau giữa hai lần (greedy decode, không train). Như vậy dao động do huấn
  luyện fp16 trên GPU là khoảng ±0.015 target — đủ để thứ tự `correct` vs `attn_only` đảo chiều
  giữa hai lần, nhưng không đủ để đổi phán quyết.
* **n = 50** mẫu eval: một mẫu tương đương 0.02 điểm. Các chênh lệch dưới ~0.03 nên được đọc
  là "không phân biệt được".
* Corpus mặc định được chấm bằng `triage_field_accuracy`, ViSFD bằng `aspect_f1` — **không**
  so trực tiếp con số target giữa hai corpus, chỉ so Δ so với (b) của từng corpus.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [x] **B2 dataset miền riêng** — UIT-ViSFD, 250 train + 50 eval, `data/CUSTOM_DATASET.md`
  (nguồn, bộ lọc, khử nhiễm 4 lớp, lý do mới về phân phối), dựng lại được bằng
  `python scripts/make_visfd_data.py`
- [ ] B3 reasoning-trace collapse — không làm: dữ liệu không có reasoning trace, `MASK_MODE`
  khác nhau sẽ cho cùng một mask (xem `.env.example`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
