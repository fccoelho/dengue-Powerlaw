const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 9,
  title: 'Model Validation'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  slide.addText("Model Validation", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 32, fontFace: "Arial",
    color: theme.primary, bold: true
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.95, w: 2, h: 0.05,
    fill: { color: theme.accent }
  });

  slide.addText("Power Law vs Exponential Comparison", {
    x: 0.5, y: 1.15, w: 9, h: 0.4,
    fontSize: 14, fontFace: "Arial",
    color: theme.secondary, italic: true
  });

  slide.addImage({
    x: 0.3, y: 1.5, w: 9.2, h: 3.5,
    path: "./imgs/model_comparison.png"
  });

  const legend = () => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 6.5, y: 4.7, w: 3.0, h: 0.7,
      fill: { color: theme.primary },
      rectRadius: 0.08
    });
    slide.addText("R > 0 & p ≤ 0.05 → Power law preferred", {
      x: 6.6, y: 4.75, w: 2.8, h: 0.6,
      fontSize: 10, fontFace: "Arial",
      color: "FFFFFF", align: "center", valign: "middle"
    });
  };
  legend();

  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent }
  });
  slide.addText("9", {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });

  return slide;
}

if (require.main === module) {
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_16x9';
  const theme = {
    primary: "1A365D",
    secondary: "2C5282",
    accent: "4ECDC4",
    light: "E2E8F0",
    bg: "F7FAFC"
  };
  createSlide(pres, theme);
  pres.writeFile({ fileName: "slide-09-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
