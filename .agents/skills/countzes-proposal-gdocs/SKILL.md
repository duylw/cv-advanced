---
name: countzes-proposal-gdocs
description: >-
  Use this skill whenever creating or updating a multi-tab Google Doc to document
  improvement proposals (proposes) in CountZES: Tab 1 is Overview, each improvement
  proposal has its own clearly named tab, written in clear conceptual/non-code prose
  without code snippets or CLI/hyperparameter clutter.
---

# CountZES Proposal Google Docs Skill

Skill này hướng dẫn cách tạo và cập nhật **Google Doc nhiều tab** để trình bày các **đề xuất cải tiến (proposals)** cho dự án **CountZES** theo văn phong dễ hiểu, định hướng báo cáo, hạn chế tối đa code và tham số kỹ thuật.

---

## 1. Khi nào kích hoạt Skill
- Khi người dùng muốn ghi lại, tổng hợp, hoặc cập nhật một hoặc nhiều **đề xuất cải tiến mô hình / pipeline** lên Google Docs.
- Khi có thêm một kết quả thử nghiệm cải tiến mới cần bổ sung vào tài liệu đề xuất trên Google Docs (thêm tab đề xuất mới và cập nhật tab `Overview`).

---

## 2. Nguyên tắc viết nội dung (Bắt buộc tuân thủ)

1. **Hạn chế nói về code:**
   - **Không** đưa code block lập trình (Python, bash, CLI command), tên file mã nguồn (`.py`, `.ipynb`), tên hàm, tên biến, hay chi tiết thao tác tensor (`[:, 1:, :]`, tên API thư viện).
   - Nếu cần mô tả quy trình, chỉ dùng lời văn tự nhiên hoặc sơ đồ ý niệm ngắn gọn bằng chữ.
2. **Hạn chế nhắc tham số & cờ cấu hình:**
   - **Không** liệt kê tên cờ dòng lệnh (`--clip_sim_mode`, `--box_threshold`, ...), không nhồi các con số ngưỡng kỹ thuật cứng (`0.15`, `p=90`, `seed=42`...) vào phần giải thích ý tưởng.
   - Tập trung diễn đạt **ý nghĩa trực quan**: mô hình đang nhìn sai chỗ nào, vì sao bị vậy, và ý tưởng mới chỉnh lại luồng suy nghĩ của mô hình ra sao.
3. **Văn phong dễ hiểu, hướng người đọc báo cáo:**
   - Viết như giải thích cho người đọc báo cáo hoặc giảng viên hướng dẫn nắm ngay bản chất vấn đề trong 2–3 phút.
   - Ưu tiên ví dụ trực quan trên ảnh thật (ví dụ: *“mô hình nhầm sang đống rơm bên cạnh thay vì giỏ táo”*) kèm hình ảnh đối chiếu Trước / Sau khi có sẵn.

---

## 3. Cấu trúc Multi-Tab chuẩn & Khả năng mở rộng (Scalable Design)

Tài liệu phải **dễ thêm mới, dễ loại bỏ, và không phụ thuộc cứng vào số thứ tự** khi các thử nghiệm thay đổi:

### Quy tắc đặt tên & Vòng đời Tab
1. **Không gắn số thứ tự cứng (`1.`, `2.`, `3.`) vào tên tab:** Dùng tên ngắn theo đúng chủ đề đề xuất (ví dụ: `MaskCLIP & Giữ khung`, `Khử trùng lặp (SDES)`). Khi cần loại bỏ 1 đề xuất hoặc chèn 1 đề xuất mới, các tab khác không bị lệch số thứ tự.
2. **Gắn nhãn trạng thái ngay trên tên hoặc đầu tab:**
   - Đề xuất chưa chạy thực nghiệm: `[Nháp] <Tên đề xuất>`
   - Đề xuất đang test / đã chốt giữ: `<Tên đề xuất>`
   - Đề xuất test không hiệu quả nhưng giữ lại làm bài học cho báo cáo: `[Loại] <Tên đề xuất>`
3. **Create-on-Demand & Template Placeholder:**
   - Ý tưởng mới manh nha chỉ cần ghi 1 dòng vào bảng **Backlog / Placeholder** trong tab `Overview`; chỉ tạo tab riêng khi bắt đầu thiết kế chi tiết hoặc chạy thực nghiệm.
   - Duy trì sẵn 1 tab **`[Template] Đề xuất mới`** chứa khung 5 mục có sẵn các ô `[Placeholder]` để sao chép/điền nhanh khi thêm đề xuất mới.
4. **Mỗi Tab tự đứng độc lập (Self-contained):** Nội dung bên trong 1 tab đề xuất không tham chiếu cứng *"như Đề xuất số 2 ở trên"* để có thể ẩn, xóa hoặc sắp xếp lại bất kỳ lúc nào.

### Tab 1: `Overview`
Tab đầu tiên **luôn là phần tổng quan**, gồm:
1. **Bối cảnh & Mục tiêu:** Pipeline gốc đang làm gì (3 bước từ khoanh vùng ban đầu → tinh chỉnh theo mật độ → chọn đại diện đồng thuận) và bức tranh các nhóm vấn đề lớn.
2. **Bảng tổng hợp các đề xuất (có cột Trạng thái & dòng Placeholder sẵn):**
   | Tên đề xuất (Tab) | Vấn đề cốt lõi giải quyết | Giai đoạn tác động | Trạng thái | Hiệu quả chính |
   | :--- | :--- | :--- | :--- | :--- |
   *(Bao gồm 1 dòng `[Placeholder — Đề xuất tiếp theo]` làm mẫu mở rộng; tuyệt đối không thêm cột tham số kỹ thuật hay cờ dòng lệnh.)*
3. **Tổng hợp kết quả thực nghiệm (nếu đã có):** Bảng so sánh ngắn gọn các chỉ số đếm chính (`MAE`, `RMSE`, tốc độ tương đối) giữa Baseline và các đề xuất đã chạy.
4. **Backlog / Ý tưởng mở (Tùy chọn):** Danh sách ngắn các điểm yếu mới phát hiện từ thực nghiệm chưa tách thành tab riêng để tiện cân nhắc thêm hoặc bỏ.

### Các Tab đề xuất & Tab `[Template] Đề xuất mới`
- **Khung nội dung chuẩn cho mỗi Tab đề xuất (5 mục gọn, kèm placeholder khi chưa có dữ liệu):**
  1. **Vấn đề gặp phải:** Mô tả trực quan lỗi hoặc hạn chế của mô hình gốc trên ảnh thực tế.
  2. **Nguyên nhân cốt lõi:** Giải thích ngắn gọn, dễ hiểu tại sao cơ chế cũ lại dẫn đến lỗi đó.
  3. **Ý tưởng đề xuất:** Trình bày giải pháp mới giải quyết đúng nguyên nhân vừa nêu như thế nào (giữ ở mức ý niệm, không chép code).
  4. **Kết quả & Quan sát thực tế:** So sánh ngắn gọn trước/sau (bảng số liệu + ảnh đối chiếu nếu đã chạy; nếu chưa chạy thì giữ placeholder `"[Chờ chạy thực nghiệm đối chứng với Baseline]"`).
  5. **Ưu điểm & Điểm cần lưu ý:** Nhận xét khách quan về lợi ích, giới hạn còn lại và quyết định (`Giữ lại` / `Cần tinh chỉnh` / `Loại bỏ`).

---

## 4. Quy trình thao tác Google Docs Multi-Tab (`gdocs` CLI)

Tuân thủ skill `gdocs` (đã cài từ plugin Google Docs):

1. **Chuẩn bị nội dung Markdown sạch trong thư mục scratch** (`<appDataDir>/brain/<conversation-id>/scratch/`), tuyệt đối không để file nháp trong workspace dự án:
   - Chỉ dùng cú pháp mà `gdocs` hỗ trợ tốt: `#`, `##`, `###`, `**đậm**`, `*nghiêng*`, danh sách, bảng chuẩn `| ... |`.
   - **Tránh** blockquote (`>`), cảnh báo GitHub (`[!NOTE]`), công thức LaTeX (`$...$`), HTML thô (vì sẽ bị biến thành chữ thường khi import/sync). Dùng ký tự Unicode (`×`, `→`, `≈`) thay cho LaTeX.
2. **Nếu tạo Google Doc hoàn toàn mới:**
   - Import tab `Overview` trước để tạo Doc:
     ```bash
     gdocs mutate import-md overview.md --title "CountZES - Tổng hợp Đề xuất Cải tiến" --auto-fit-tables
     ```
   - Lấy danh sách tab hiện có (`DOC_ID` từ lệnh trên) và đổi tên tab mặc định đầu tiên thành `Overview`:
     ```bash
     gdocs readonly list-tabs DOC_ID
     gdocs mutate rename-tab DOC_ID <FIRST_TAB_ID> "Overview"
     ```
3. **Thêm (hoặc cập nhật) Tab cho một đề xuất mới:**
   - Kiểm tra `gdocs readonly list-tabs DOC_ID` xem tab của đề xuất đó đã có chưa:
     - Nếu chưa có, tạo tab mới với tên rõ ràng:
       ```bash
       gdocs mutate create-tab DOC_ID --title "1. Tên đề xuất ngắn gọn"
       ```
     - Lấy `<TAB_ID>` của tab vừa tạo từ output (hoặc `readonly list-tabs DOC_ID`).
   - Đồng bộ nội dung từ file Markdown đề xuất vào đúng `<TAB_ID>` và căn chỉnh bảng:
     ```bash
     gdocs mutate sync DOC_ID proposal_1.md --tab <TAB_ID> --prefer-local
     gdocs mutate auto-fit-tables DOC_ID --tab <TAB_ID>
     ```
   - *(Nếu cần cập nhật lại tab `Overview` khi thêm đề xuất mới, chạy `gdocs mutate sync DOC_ID overview.md --tab <OVERVIEW_TAB_ID> --prefer-local` — lưu ý kiểm tra xem tài liệu có comment mở từ người dùng chưa trước khi ghi đè).*
4. **Chèn ảnh minh họa Trước / Sau (nếu có):**
   - Dùng `gdocs mutate insert-image-from-file DOC_ID <path_anh> --tab <TAB_ID> --after "<đoạn chữ neo>"`.
   - Không truyền `--alt-title` nếu tài khoản gặp giới hạn *suggestion mode*; nếu chèn nhiều ảnh sau cùng 1 đoạn neo, chèn ảnh cuối trước để thứ tự hiển thị đúng từ trên xuống dưới.
5. **Xác nhận & Trả link cho người dùng:**
   - Đọc lại tab vừa cập nhật bằng `gdocs readonly read DOC_ID --tab <TAB_ID>` và `gdocs readonly list-images DOC_ID --tab <TAB_ID>` để chắc chắn không lỗi định dạng.
   - Lấy link trực tiếp tới tài liệu và tab vừa sửa bằng `gdocs readonly get-url DOC_ID --tab <TAB_ID>` và gửi link clickable Markdown trong câu trả lời.
