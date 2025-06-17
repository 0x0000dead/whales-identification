import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';
import * as api from '../api';

jest.mock('../api');

const mockSingle = api.predictSingle as jest.Mock;
const mockBatch  = api.predictBatch  as jest.Mock;

describe('<App />', () => {
  beforeEach(() => {
    mockSingle.mockReset();
    mockBatch.mockReset();
  });

  it('показывает результат single-предикта', async () => {
    const fake = {
      image_ind: 'img.png',
      bbox: [0,0,5,5],
      class_animal: 'whale',
      id_animal: 'W-001',
      probability: 0.99,
      mask: undefined,
    };
    mockSingle.mockResolvedValue(fake);

    const { container } = render(<App />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const btn       = screen.getByText(/Send|Отправить/i);

    // эмулируем выбор файла
    const file = new File([''], 'img.png', { type: 'image/png' });
    fireEvent.change(fileInput, { target: { files: [file] } });

    fireEvent.click(btn);

    // ждём, что JSON-результат отобразится
    await waitFor(() => {
      expect(screen.getByText(/"image_ind": "img.png"/)).toBeInTheDocument();
      expect(screen.getByText(/"probability": 0.99/)).toBeInTheDocument();
    });
  });

  it('показывает дашборд после batch-предикта', async () => {
    // 2 типа китов: A×2, B×1
    mockBatch.mockResolvedValue([
      { image_ind:'1.png', bbox:[0,0,1,1], class_animal:'whale', id_animal:'A', probability:0.9 },
      { image_ind:'2.png', bbox:[0,0,1,1], class_animal:'whale', id_animal:'A', probability:0.8 },
      { image_ind:'3.png', bbox:[0,0,1,1], class_animal:'whale', id_animal:'B', probability:0.85 },
    ]);

    const { container } = render(<App />);
    const inputs = container.querySelectorAll('input[type="file"]');
    const batchInput = inputs[1];
    const btn = screen.getByText(/Run Batch|Скачать CSV/i);

    const zip = new File([''], 'batch.zip', { type: 'application/zip' });
    fireEvent.change(batchInput, { target: { files: [zip] } });
    fireEvent.click(btn);

    // ждём заголовок дашборда
    await waitFor(() => {
      expect(screen.getByText(/Whale Types Distribution/i)).toBeInTheDocument();
    });

    // проверим, что два разных имени китов попали в chartData
    // в Recharts имена отображаются как текст на оси X
    await waitFor(() => {
      expect(screen.getByText('A')).toBeInTheDocument();
      expect(screen.getByText('B')).toBeInTheDocument();
    });
  });
});
