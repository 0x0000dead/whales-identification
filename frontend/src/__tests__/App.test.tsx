import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';
import * as api from '../api';

jest.mock('../api');

const mockSingle  = api.predictSingle  as jest.Mock;
const mockBatch   = api.predictBatch   as jest.Mock;
const mockHealth  = api.checkHealth    as jest.Mock;

// Default: server reports healthy so the indicator doesn't distract other tests
beforeEach(() => {
  mockSingle.mockReset();
  mockBatch.mockReset();
  mockHealth.mockResolvedValue(true);
});

// ── Helpers ─────────────────────────────────────────────────────────────────

function fakeDetection(overrides: Partial<api.Detection> = {}): api.Detection {
  return {
    image_ind: 'img.png',
    bbox: [0, 0, 100, 100],
    class_animal: '1a71fbb72250',
    id_animal: 'humpback_whale',
    probability: 0.92,
    is_cetacean: true,
    cetacean_score: 0.91,
    rejected: false,
    rejection_reason: null,
    model_version: 'stub-v1',
    candidates: [
      { class_animal: 'abc456', id_animal: 'killer_whale',        probability: 0.55 },
      { class_animal: 'cafe09', id_animal: 'bottlenose_dolphin',  probability: 0.28 },
    ],
    ...overrides,
  };
}

// ── Server indicator ─────────────────────────────────────────────────────────

describe('Server status indicator', () => {
  it('shows "Сервер готов" when health check returns true', async () => {
    mockHealth.mockResolvedValue(true);
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Сервер готов/i)).toBeInTheDocument();
    });
  });

  it('shows "Сервер недоступен" when health check returns false', async () => {
    mockHealth.mockResolvedValue(false);
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Сервер недоступен/i)).toBeInTheDocument();
    });
  });
});

// ── Single upload ─────────────────────────────────────────────────────────────

describe('Single image upload', () => {
  it('shows species and individual ID after accepted prediction', async () => {
    const fake = fakeDetection({ id_animal: 'W-001', probability: 0.99 });
    mockSingle.mockResolvedValue(fake);

    const { container } = render(<App />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const btn = screen.getByText(/Определить вид и особь/i);

    fireEvent.change(fileInput, { target: { files: [new File([''], 'img.png', { type: 'image/png' })] } });
    fireEvent.click(btn);

    // The individual ID appears in both the heading and the subtitle — use getAllByText
    await waitFor(() => {
      expect(screen.getAllByText(/W-001/).length).toBeGreaterThan(0);
    });
  });

  it('shows alternative candidates section after accepted prediction', async () => {
    const fake = fakeDetection();
    mockSingle.mockResolvedValue(fake);

    const { container } = render(<App />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;

    fireEvent.change(fileInput, { target: { files: [new File([''], 'img.png', { type: 'image/png' })] } });
    fireEvent.click(screen.getByText(/Определить вид и особь/i));

    await waitFor(() => {
      expect(screen.getByText(/Альтернативные варианты/i)).toBeInTheDocument();
    });
  });

  it('shows RejectionCard "не является морским млекопитающим" for not_a_marine_mammal', async () => {
    const fake = fakeDetection({
      rejected: true,
      is_cetacean: false,
      rejection_reason: 'not_a_marine_mammal',
      probability: 0.0,
      candidates: [],
    });
    mockSingle.mockResolvedValue(fake);

    const { container } = render(<App />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;

    fireEvent.change(fileInput, { target: { files: [new File([''], 'photo.png', { type: 'image/png' })] } });
    fireEvent.click(screen.getByText(/Определить вид и особь/i));

    await waitFor(() => {
      expect(screen.getByText(/Не является морским млекопитающим/i)).toBeInTheDocument();
    });
  });

  it('shows RejectionCard "особь не найдена" for low_confidence', async () => {
    const fake = fakeDetection({
      rejected: true,
      is_cetacean: true,
      rejection_reason: 'low_confidence',
      probability: 0.02,
    });
    mockSingle.mockResolvedValue(fake);

    const { container } = render(<App />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;

    fireEvent.change(fileInput, { target: { files: [new File([''], 'photo.png', { type: 'image/png' })] } });
    fireEvent.click(screen.getByText(/Определить вид и особь/i));

    await waitFor(() => {
      expect(screen.getByText(/Особь не найдена в базе данных/i)).toBeInTheDocument();
    });
  });

  it('shows mailto link in RejectionCard', async () => {
    const fake = fakeDetection({
      rejected: true,
      rejection_reason: 'low_confidence',
      candidates: [],
    });
    mockSingle.mockResolvedValue(fake);

    const { container } = render(<App />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;

    fireEvent.change(fileInput, { target: { files: [new File([''], 'photo.png', { type: 'image/png' })] } });
    fireEvent.click(screen.getByText(/Определить вид и особь/i));

    await waitFor(() => {
      const mailLink = container.querySelector('a[href^="mailto:vandanov2010@gmail.com"]');
      expect(mailLink).not.toBeNull();
    });
  });
});

// ── Batch upload ─────────────────────────────────────────────────────────────

describe('Batch upload', () => {
  it('shows batch dashboard after successful batch prediction', async () => {
    mockBatch.mockResolvedValue([
      fakeDetection({ image_ind: '1.png', id_animal: 'A' }),
      fakeDetection({ image_ind: '2.png', id_animal: 'A' }),
      fakeDetection({ image_ind: '3.png', id_animal: 'B' }),
    ]);

    const { container } = render(<App />);

    // Switch to batch tab and wait for re-render
    fireEvent.click(screen.getByText(/Пакетная обработка/i));
    await waitFor(() =>
      expect(screen.getByText(/Обработать архив/i)).toBeInTheDocument()
    );

    // There is now only 1 file input (batch tab)
    const batchInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(batchInput, { target: { files: [new File([''], 'batch.zip', { type: 'application/zip' })] } });
    fireEvent.click(screen.getByText(/Обработать архив/i));

    await waitFor(() => {
      expect(screen.getByText(/Распределение по видам/i)).toBeInTheDocument();
    }, { timeout: 3000 });
  });
});
