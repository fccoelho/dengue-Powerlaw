const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 5,
  title: 'Power Law Methodology'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  slide.addText("Power Law Fitting Methodology", {
    x: 0.5, y: 0.3, w: 9, h: 0.8,
    fontSize: 32, fontFace: "Arial",
    color: theme.primary, bold: true
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.0, w: 2, h: 0.05,
    fill: { color: theme.accent }
  });

  slide.addText("Following Clauset et al. (2009) maximum likelihood approach", {
    x: 0.5, y: 1.2, w: 9, h: 0.4,
    fontSize: 14, fontFace: "Arial",
    color: theme.secondary, italic: true
  });

  const formulaBox = () => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.5, y: 1.7, w: 4.3, h: 1.8,
      fill: { color: theme.primary },
      rectRadius: 0.1
    });
    slide.addText("P(X ≥ x) ∝ x⁻⁽ᵅ⁻¹⁾", {
      x: 0.7, y: 1.9, w: 4, h: 0.7,
      fontSize: 24, fontFace: "Arial",
      color: "FFFFFF", bold: true, align: "center"
    });
    slide.addText("Power law probability distribution\nwhere α is the scaling exponent", {
      x: 0.7, y: 2.6, w: 4, h: 0.7,
      fontSize: 12, fontFace: "Arial",
      color: theme.light, align: "center"
    });
  };
  formulaBox();

  const paramsBox = () => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 5.2, y: 1.7, w: 4.3, h: 1.8,
      fill: { color: "FFFFFF" },
      line: { color: theme.accent, width: 2 },
      rectRadius: 0.1
    });
    slide.addText("Key Parameters", {
      x: 5.4, y: 1.85, w: 4, h: 0.35,
      fontSize: 14, fontFace: "Arial",
      color: theme.primary, bold: true
    });
    slide.addText([
      { text: "α (alpha): ", options: { bold: true } },
      { text: "scaling exponent", options: { breakLine: true } },
      { text: "xmin: ", options: { bold: true } },
      { text: "tail begins", options: { breakLine: true } },
      { text: "xmax: ", options: { bold: true } },
      { text: "data limit", options: { breakLine: true } },
      { text: "R, p: ", options: { bold: true } },
      { text: "power law vs exponential test", options: {} }
    ], {
      x: 5.4, y: 2.25, w: 4, h: 1.1,
      fontSize: 12, fontFace: "Arial",
      color: theme.secondary
    });
  };
  paramsBox();

  slide.addText("Fitting Levels", {
    x: 0.5, y: 3.7, w: 9, h: 0.4,
    fontSize: 16, fontFace: "Arial",
    color: theme.primary, bold: true
  });

  const levels = [
    { level: "City", desc: "Weekly cases per municipality", color: theme.accent },
    { level: "Yearly", desc: "Annual fits per city", color: theme.secondary },
    { level: "State", desc: "Aggregated state-level fits", color: theme.primary },
    { level: "National", desc: "Brazil-wide distributions", color: theme.primary }
  ];

  levels.forEach((l, i) => {
    const x = 0.5 + i * 2.35;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x, y: 4.15, w: 2.2, h: 0.9,
      fill: { color: l.color },
      rectRadius: 0.08
    });
    slide.addText(l.level, {
      x: x, y: 4.2, w: 2.2, h: 0.35,
      fontSize: 13, fontFace: "Arial",
      color: "FFFFFF", bold: true, align: "center"
    });
    slide.addText(l.desc, {
      x: x, y: 4.5, w: 2.2, h: 0.45,
      fontSize: 9, fontFace: "Arial",
      color: theme.light, align: "center"
    });
  });

  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent }
  });
  slide.addText("5", {
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
  pres.writeFile({ fileName: "slide-05-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
