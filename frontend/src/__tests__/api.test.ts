import { predictSingle, predictBatch, checkHealth, Detection, Candidate } from '../api';

describe('API helpers', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  // ── predictSingle ─────────────────────────────────────────────────────────

  it('predictSingle: calls fetch and returns parsed JSON', async () => {
    const fake: Detection = {
      image_ind: 'img.png',
      bbox: [1, 2, 3, 4],
      class_animal: 'whale',
      id_animal: 'humpback_whale',
      probability: 0.95,
      mask: 'base64str',
    };

    (global.fetch as jest.Mock).mockResolvedValue({ ok: true, json: async () => fake });

    const file = new File([''], 'img.png', { type: 'image/png' });
    const result = await predictSingle(file);

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/predict-single'),
      expect.objectContaining({ method: 'POST', body: expect.any(FormData) })
    );
    expect(result).toEqual(fake);
  });

  it('predictSingle: Detection type accepts optional candidates field', () => {
    const candidates: Candidate[] = [
      { class_animal: 'abc', id_animal: 'killer_whale', probability: 0.6 },
    ];
    const det: Detection = {
      image_ind: 'img.png',
      bbox: [0, 0, 100, 100],
      class_animal: 'whale',
      id_animal: 'humpback_whale',
      probability: 0.9,
      candidates,
    };
    expect(det.candidates).toHaveLength(1);
    expect(det.candidates![0].id_animal).toBe('killer_whale');
  });

  it('predictSingle: throws network error with helpful message on fetch failure', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(new TypeError('Failed to fetch'));

    const file = new File([''], 'img.png', { type: 'image/png' });
    await expect(predictSingle(file)).rejects.toThrow(/Сервер анализа недоступен/);
  });

  it('predictSingle: throws on non-ok HTTP response', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 415,
      text: async () => 'Unsupported Media Type',
    });

    const file = new File([''], 'img.png', { type: 'image/png' });
    await expect(predictSingle(file)).rejects.toThrow(/415/);
  });

  // ── predictBatch ──────────────────────────────────────────────────────────

  it('predictBatch: returns array of detections', async () => {
    const fakeArr: Detection[] = [
      { image_ind: 'a.png', bbox: [0,0,1,1], class_animal: 'whale', id_animal: 'W1', probability: 0.9 },
      { image_ind: 'b.png', bbox: [1,1,2,2], class_animal: 'whale', id_animal: 'W1', probability: 0.88 },
    ];
    (global.fetch as jest.Mock).mockResolvedValue({ ok: true, json: async () => fakeArr });

    const zip = new File([''], 'batch.zip', { type: 'application/zip' });
    const data = await predictBatch(zip);

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/predict-batch'),
      expect.objectContaining({ method: 'POST', body: expect.any(FormData) })
    );
    expect(data).toEqual(fakeArr);
  });

  // ── checkHealth ───────────────────────────────────────────────────────────

  it('checkHealth: returns true when server responds 200', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({ ok: true });
    expect(await checkHealth()).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/health'),
      expect.objectContaining({ signal: expect.anything() })
    );
  });

  it('checkHealth: returns false when server responds non-200', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({ ok: false });
    expect(await checkHealth()).toBe(false);
  });

  it('checkHealth: returns false on network error (no crash)', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(new TypeError('Network failure'));
    expect(await checkHealth()).toBe(false);
  });
});
