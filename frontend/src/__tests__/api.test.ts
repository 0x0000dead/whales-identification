import { predictSingle, predictBatch, Detection } from '../api';

describe('API helpers', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  it('predictSingle: вызывает fetch и возвращает распарсенный JSON', async () => {
    const fake: Detection = {
      image_ind: 'img.png',
      bbox: [1, 2, 3, 4],
      class_animal: 'whale',
      id_animal: 'Горбатый кит',
      probability: 0.95,
      mask: 'base64str',
    };

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => fake,
    });

    const file = new File([''], 'img.png', { type: 'image/png' });
    const result = await predictSingle(file);

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/predict-single'),
      expect.objectContaining({ method: 'POST', body: expect.any(FormData) })
    );
    expect(result).toEqual(fake);
  });

  it('predictBatch: возвращает массив детекций', async () => {
    const fakeArr: Detection[] = [
      { image_ind: 'a.png', bbox: [0,0,1,1], class_animal: 'whale', id_animal: 'W1', probability: 0.9 },
      { image_ind: 'b.png', bbox: [1,1,2,2], class_animal: 'whale', id_animal: 'W1', probability: 0.88 },
    ];
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => fakeArr,
    });

    const zip = new File([''], 'batch.zip', { type: 'application/zip' });
    const data = await predictBatch(zip);

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/predict-batch'),
      expect.objectContaining({ method: 'POST', body: expect.any(FormData) })
    );
    expect(data).toEqual(fakeArr);
  });
});
