import { useEffect, useRef, useState } from "react";
import { confidence, fileUrl, request, statusText } from "./api";
import type { Health, Job, JobPage, MediaType } from "./api";

function Icon({
  name,
}: {
  name: "shield" | "upload" | "image" | "video" | "arrow" | "clock";
}) {
  const paths = {
    shield: (
      <>
        <path d="M12 3 4 6v6c0 5 8 9 8 9s8-4 8-9V6z" />
        <path d="m8 12 3 3 5-6" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5" />
      </>
    ),
    image: (
      <>
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <circle cx="8" cy="8" r="1.5" />
        <path d="m3 17 6-6 5 5 3-3 4 4" />
      </>
    ),
    video: (
      <>
        <rect x="3" y="5" width="18" height="14" rx="2" />
        <path d="m10 9 5 3-5 3z" />
      </>
    ),
    arrow: (
      <>
        <path d="M4 12h16m-6-6 6 6-6 6" />
      </>
    ),
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
  };
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}
function Media({
  url,
  kind,
  alt,
}: {
  url: string;
  kind: MediaType;
  alt: string;
}) {
  return kind === "image" ? (
    <img className="media" src={url} alt={alt} />
  ) : (
    <video
      className="media"
      src={url}
      controls
      preload="metadata"
      aria-label={alt}
    />
  );
}
function Results({ job }: { job: Job }) {
  const done = job.status === "completed";
  return (
    <section className="result panel" aria-labelledby="result-title">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">KẾT QUẢ NHẬN DIỆN</span>
          <h2 id="result-title">{job.original_filename}</h2>
        </div>
        <span className={`badge ${job.status}`}>{statusText[job.status]}</span>
      </div>
      {!done && (
        <div className="job-state" aria-live="polite">
          <Icon name={job.status === "failed" ? "shield" : "clock"} />
          <h3>{statusText[job.status]}</h3>
          {job.status === "failed" ? (
            <p role="alert">{job.error_message}</p>
          ) : (
            <>
              <p>
                {job.status === "queued"
                  ? "Tệp đã vào hàng đợi. Hệ thống xử lý lần lượt từng lượt."
                  : "Hệ thống đang nhận diện và chuẩn bị tệp kết quả."}
              </p>
              <progress
                max="100"
                value={job.progress}
                aria-label="Tiến độ nhận diện"
              />
              <strong>{job.progress}%</strong>
            </>
          )}
        </div>
      )}
      {done && (
        <>
          <div className="comparison">
            <figure>
              <figcaption>Ảnh/video gốc</figcaption>
              <Media
                url={fileUrl(job.id, false)}
                kind={job.media_type}
                alt="Tệp gốc"
              />
            </figure>
            <figure>
              <figcaption>Kết quả có chú thích</figcaption>
              <Media
                url={fileUrl(job.id, true)}
                kind={job.media_type}
                alt="Kết quả nhận diện với nhãn tiếng Việt"
              />
            </figure>
          </div>
          <div className="result-actions">
            <p>Độ tin cậy được hiển thị trên từng khung nhận diện.</p>
            <a className="button primary" href={fileUrl(job.id, true, true)}>
              Tải {job.media_type === "image" ? "ảnh" : "video"} kết quả ↓
            </a>
          </div>
          {job.media_type === "image" && (
            <div className="detections">
              <h3>Đối tượng phát hiện</h3>
              {job.detections?.length ? (
                <ul>
                  {job.detections.map((d, i) => (
                    <li key={i}>
                      <span>
                        <i className={`dot class-${d.class_id}`} />
                        {d.label}
                      </span>
                      <strong>{confidence(d.confidence)}</strong>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>Không phát hiện đối tượng ở ngưỡng hiện tại</p>
              )}
            </div>
          )}
          {job.media_type === "video" && (
            <p className="note">
              Video được xử lý theo từng frame. Demo không theo dõi hoặc đếm số
              người vi phạm.
            </p>
          )}
        </>
      )}
      <p className="job-meta">
        Kích thước suy luận: {job.imgsz} px · Ngưỡng tin cậy:{" "}
        {confidence(job.confidence_threshold)}
      </p>
    </section>
  );
}
function History({ onOpen }: { onOpen: (job: Job) => void }) {
  const [page, setPage] = useState(1),
    [data, setData] = useState<JobPage | null>(null),
    [error, setError] = useState(""),
    [version, setVersion] = useState(0);
  useEffect(() => {
    let alive = true;
    async function load() {
      try {
        const response = await request<JobPage>(`/jobs?page=${page}`);
        if (alive) {
          setData(response);
          setError("");
        }
      } catch (e) {
        if (alive) setError((e as Error).message);
      }
    }
    void load();
    const timer = setInterval(load, 5000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [page, version]);
  return (
    <section className="panel history">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">CÁC LƯỢT ĐÃ GỬI</span>
          <h2>Lịch sử nhận diện</h2>
        </div>
        <button
          className="button secondary"
          onClick={() => setVersion((v) => v + 1)}
        >
          Làm mới
        </button>
      </div>
      {error && (
        <p className="alert error" role="alert">
          {error}
        </p>
      )}
      {!data && !error && <p className="empty">Đang tải lịch sử…</p>}
      {data && (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Tên tệp</th>
                  <th>Loại</th>
                  <th>Thời gian</th>
                  <th>Trạng thái</th>
                  <th>Kết quả</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((job) => (
                  <tr key={job.id}>
                    <td className="filename">{job.original_filename}</td>
                    <td>{job.media_type === "image" ? "Ảnh" : "Video"}</td>
                    <td>
                      {new Date(job.created_at).toLocaleString("vi-VN", {
                        timeZone: "Asia/Ho_Chi_Minh",
                      })}
                    </td>
                    <td>
                      <span className={`badge ${job.status}`}>
                        {statusText[job.status]}
                      </span>
                    </td>
                    <td>
                      <div className="row-actions">
                        <button onClick={() => onOpen(job)}>
                          Xem{" "}
                          {job.status === "completed" ? "kết quả" : "chi tiết"}
                        </button>
                        {job.status === "completed" && (
                          <a href={fileUrl(job.id, true, true)}>Tải xuống</a>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.total === 0 && (
            <p className="empty">
              Chưa có lượt nhận diện. Hãy chọn một tệp để bắt đầu.
            </p>
          )}
          <div className="pagination">
            <span>
              {data.total} lượt nhận diện · Trang {page}/
              {Math.max(1, Math.ceil(data.total / data.page_size))}
            </span>
            <div>
              <button
                className="button secondary"
                disabled={page === 1}
                onClick={() => setPage((p) => p - 1)}
              >
                Trước
              </button>
              <button
                className="button secondary"
                disabled={page * data.page_size >= data.total}
                onClick={() => setPage((p) => p + 1)}
              >
                Sau
              </button>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
export default function App() {
  const [tab, setTab] = useState<"detect" | "history">("detect"),
    [kind, setKind] = useState<MediaType>("image");
  const [file, setFile] = useState<File | null>(null),
    [preview, setPreview] = useState(""),
    [job, setJob] = useState<Job | null>(null);
  const [health, setHealth] = useState<Health | null>(null),
    [healthError, setHealthError] = useState(""),
    [error, setError] = useState(""),
    [sending, setSending] = useState(false),
    [retry, setRetry] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const busy =
    sending || job?.status === "queued" || job?.status === "processing";
  useEffect(() => {
    let alive = true;
    async function load() {
      try {
        const response = await request<Health>("/health");
        if (alive) {
          setHealth(response);
          setHealthError("");
        }
      } catch {
        if (alive) {
          setHealth(null);
          setHealthError(
            "Không kết nối được backend. Kiểm tra dịch vụ tại cổng 8000.",
          );
        }
      }
    }
    void load();
    const timer = setInterval(load, 10000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => {
    if (!job || !["queued", "processing"].includes(job.status)) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await request<Job>(`/jobs/${job!.id}`);
        if (!alive) return;
        setJob(next);
        setError("");
        if (["queued", "processing"].includes(next.status))
          timer = setTimeout(poll, 1000);
      } catch (e) {
        if (alive) {
          setError((e as Error).message);
          timer = setTimeout(poll, 3000);
        }
      }
    }
    timer = setTimeout(poll, 1000);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [job?.id, job?.status, retry]);
  function choose(next: File | undefined) {
    if (!next) return;
    setError("");
    const allowed = kind === "image" ? /\.(jpe?g|png)$/i : /\.mp4$/i;
    if (!allowed.test(next.name)) {
      setError(
        "Định dạng không phù hợp. Chọn JPG/PNG cho ảnh hoặc MP4 cho video.",
      );
      return;
    }
    if (next.size > (health?.max_upload_mb ?? 1024) * 1024 * 1024) {
      setError("Tệp vượt dung lượng cho phép.");
      return;
    }
    setFile(next);
    setJob(null);
  }
  async function submit() {
    if (!file || busy) return;
    setSending(true);
    setError("");
    const body = new FormData();
    body.append("file", file);
    body.append("media_type", kind);
    try {
      setJob(await request<Job>("/jobs", { method: "POST", body }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  }
  const ready =
    health?.database_ready &&
    health.model_ready &&
    health.font_ready &&
    (kind !== "video" || health.ffmpeg_ready);
  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="brand-icon">
            <Icon name="shield" />
          </span>
          <div>
            <strong>Nhận diện mũ bảo hiểm</strong>
            <span>YOLO26s + SCConv · Bản demo</span>
          </div>
        </div>
        <span
          className={`connection ${health?.model_ready && health.database_ready ? "online" : ""}`}
        >
          <i />
          {health?.model_ready && health.database_ready
            ? "Hệ thống sẵn sàng"
            : "Đang kiểm tra hệ thống"}
        </span>
      </header>
      <main>
        <div className="page-heading">
          <div>
            <span className="eyebrow">ỨNG DỤNG THỊ GIÁC MÁY TÍNH</span>
            <h1>
              Nhận diện người đi xe máy
              <br />
              không đội mũ bảo hiểm
            </h1>
            <p>
              Tải ảnh hoặc video để xem kết quả nhận diện với chú thích tiếng
              Việt.
            </p>
          </div>
          <div className="model-card">
            <Icon name="shield" />
            <div>
              <strong>YOLO26s + SCConv</strong>
              <span>
                {health?.imgsz ?? 768} px ·{" "}
                {health?.device === "cpu"
                  ? "Xử lý bằng CPU"
                  : health?.device
                    ? "Xử lý bằng GPU"
                    : "Đang tải cấu hình"}
              </span>
            </div>
          </div>
        </div>
        <nav className="tabs" aria-label="Điều hướng chính">
          <button
            aria-current={tab === "detect" ? "page" : undefined}
            onClick={() => setTab("detect")}
          >
            <Icon name="image" />
            Nhận diện
          </button>
          <button
            aria-current={tab === "history" ? "page" : undefined}
            onClick={() => setTab("history")}
          >
            <Icon name="clock" />
            Lịch sử
          </button>
        </nav>
        {healthError && (
          <p className="alert error" role="alert">
            {healthError}
          </p>
        )}
        {health && !health.database_ready && (
          <p className="alert warning">
            MySQL chưa sẵn sàng. Kiểm tra cấu hình DB_* trong .env và khởi động
            lại backend.
          </p>
        )}
        {health && !health.model_ready && (
          <p className="alert warning">
            {health.startup_complete
              ? "Chưa tải được mô hình"
              : "Đang tải mô hình"}{" "}
            · {health.model_message}
          </p>
        )}
        {health && !health.font_ready && (
          <p className="alert warning">
            Thiếu font Unicode. Kiểm tra FONT_PATH trong .env.
          </p>
        )}
        {tab === "history" ? (
          <History
            onOpen={(next) => {
              setJob(next);
              setKind(next.media_type);
              setFile(null);
              setError("");
              setTab("detect");
            }}
          />
        ) : (
          <>
            <section className="workspace">
              <div className="panel upload-panel">
                <div className="panel-heading">
                  <div>
                    <span className="eyebrow">BƯỚC 01</span>
                    <h2>Chọn tệp đầu vào</h2>
                  </div>
                  <span className="small muted">
                    Tối đa {(health?.max_upload_mb ?? 1024) === 1024 ? "1 GB" : `${health?.max_upload_mb ?? 1024} MB`}
                  </span>
                </div>
                <div className="media-selector" aria-label="Loại tệp">
                  {(["image", "video"] as const).map((type) => (
                    <button
                      key={type}
                      aria-pressed={kind === type}
                      disabled={busy}
                      onClick={() => {
                        setKind(type);
                        setFile(null);
                        setJob(null);
                        setError("");
                        if (input.current) input.current.value = "";
                      }}
                    >
                      <Icon name={type} />
                      {type === "image" ? "Ảnh" : "Video"}
                    </button>
                  ))}
                </div>
                <input
                  ref={input}
                  id="media-file"
                  aria-label="Chọn ảnh hoặc video"
                  type="file"
                  accept={kind === "image" ? ".jpg,.jpeg,.png" : ".mp4"}
                  disabled={busy}
                  onChange={(event) => choose(event.target.files?.[0])}
                />
                <div
                  className={`dropzone ${file ? "has-file" : ""}`}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    if (!busy) choose(e.dataTransfer.files[0]);
                  }}
                >
                  {preview ? (
                    <Media
                      url={preview}
                      kind={kind}
                      alt="Xem trước tệp đã chọn"
                    />
                  ) : (
                    <>
                      <span className="upload-icon">
                        <Icon name="upload" />
                      </span>
                      <h3>
                        {kind === "image"
                          ? "Thêm ảnh cần nhận diện"
                          : "Thêm video cần nhận diện"}
                      </h3>
                      <p>Kéo thả tệp vào đây hoặc chọn từ máy tính</p>
                    </>
                  )}
                  <button
                    className="button secondary"
                    disabled={busy}
                    onClick={() => input.current?.click()}
                  >
                    {file ? "Chọn tệp khác" : "Chọn tệp"}
                  </button>
                  <span className="small muted">
                    {file
                      ? file.name
                      : kind === "image"
                        ? "Hỗ trợ JPG, JPEG, PNG"
                        : "Hỗ trợ video MP4"}
                  </span>
                </div>
                {kind === "video" && health && !health.ffmpeg_ready && (
                  <p className="alert warning">
                    Thiếu FFmpeg để tạo video phát trên trình duyệt.
                  </p>
                )}
                {error && (
                  <div className="alert error" role="alert">
                    {error}
                    {job && busy && (
                      <button
                        className="text-button"
                        onClick={() => setRetry((v) => v + 1)}
                      >
                        Thử kết nối lại
                      </button>
                    )}
                  </div>
                )}
                <button
                  className="button primary submit"
                  disabled={!file || busy || !ready}
                  onClick={submit}
                >
                  {sending
                    ? "Đang gửi tệp…"
                    : busy
                      ? "Đang nhận diện…"
                      : "Nhận diện"}
                  <Icon name="arrow" />
                </button>
                <p className="small muted footnote">
                  Tệp được xử lý lần lượt. Thời gian phụ thuộc kích thước tệp và
                  thiết bị.
                </p>
              </div>
              <aside className="guide">
                <span className="eyebrow">BƯỚC 02</span>
                <h2>Đọc kết quả nhận diện</h2>
                <p>Mỗi khung thể hiện một đối tượng và độ tin cậy tương ứng.</p>
                <ul className="legend">
                  <li>
                    <i className="dot class-0" />
                    <div>
                      <strong>Xe máy</strong>
                      <span>Phương tiện được phát hiện</span>
                    </div>
                  </li>
                  <li>
                    <i className="dot class-1" />
                    <div>
                      <strong>Người đội mũ bảo hiểm</strong>
                      <span>Khung màu xanh lá</span>
                    </div>
                  </li>
                  <li>
                    <i className="dot class-2" />
                    <div>
                      <strong>Người không đội mũ bảo hiểm</strong>
                      <span>Khung màu đỏ</span>
                    </div>
                  </li>
                </ul>
                <div className="info-note">
                  <Icon name="shield" />
                  <p>
                    Kết quả là dự đoán của mô hình và cần được kiểm tra lại. Độ
                    tin cậy không phải độ chính xác của mô hình.
                  </p>
                </div>
                <div className="guide-footer">
                  <span className="eyebrow">BƯỚC 03</span>
                  <h3>Lưu kết quả của bạn</h3>
                  <p>Tải tệp có chú thích hoặc xem lại trong mục Lịch sử.</p>
                </div>
              </aside>
            </section>
            {job && <Results job={job} />}
          </>
        )}
        <footer>
          <span>
            Đề tài nhận diện không đội mũ bảo hiểm khi điều khiển xe máy
          </span>
          <span>Ảnh · Video · Lịch sử</span>
        </footer>
      </main>
    </>
  );
}
