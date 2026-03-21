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

  it('shows single prediction result', async () => {
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
    const btn = screen.getByText(/Отправить$/i);

    const file = new File([''], 'img.png', { type: 'image/png' });
    fireEvent.change(fileInput, { target: { files: [file] } });

    fireEvent.click(btn);

    await waitFor(() => {
      expect(screen.getByText(/W-001/)).toBeInTheDocument();
      expect(screen.getByText(/0.99/)).toBeInTheDocument();
    });
  });

  it('shows batch dashboard after batch prediction', async () => {
    mockBatch.mockResolvedValue([
      { image_ind:'1.png', bbox:[0,0,1,1], class_animal:'whale', id_animal:'A', probability:0.9 },
      { image_ind:'2.png', bbox:[0,0,1,1], class_animal:'whale', id_animal:'A', probability:0.8 },
      { image_ind:'3.png', bbox:[0,0,1,1], class_animal:'whale', id_animal:'B', probability:0.85 },
    ]);

    const { container } = render(<App />);
    const inputs = container.querySelectorAll('input[type="file"]');
    const batchInput = inputs[1];
    const btn = screen.getByText(/Отправить пакет/i);

    const zip = new File([''], 'batch.zip', { type: 'application/zip' });
    fireEvent.change(batchInput, { target: { files: [zip] } });
    fireEvent.click(btn);

    await waitFor(() => {
      expect(screen.getByText(/Распределение типов сущностей/i)).toBeInTheDocument();
    });
  });
});
