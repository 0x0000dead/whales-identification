import '@testing-library/jest-dom';

// jsdom does not implement URL.createObjectURL / revokeObjectURL.
global.URL.createObjectURL = jest.fn(() => 'blob:mock-object-url');
global.URL.revokeObjectURL = jest.fn();

// jsdom does not implement ResizeObserver (used by Recharts).
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
global.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;

// jsdom does not implement SVGElement.getBBox (used by Recharts).
Object.defineProperty(window.SVGElement.prototype, 'getBBox', {
  writable: true,
  value: () => ({ x: 0, y: 0, width: 0, height: 0 }),
});
