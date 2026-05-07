import { IdAttributePlugin } from "@11ty/eleventy";
import syntaxHighlight from "@11ty/eleventy-plugin-syntaxhighlight";
import path from "node:path";

function normalizeSitePath(url) {
  if (!url || url === "/") {
    return "/";
  }

  return `/${url.replace(/^\/+/, "")}`;
}

function stripSuffix(url) {
  const match = String(url).match(/^([^?#]*)([?#].*)?$/);
  return {
    path: match?.[1] ?? "",
    suffix: match?.[2] ?? ""
  };
}

/** @param {import("@11ty/eleventy").UserConfig} eleventyConfig */
export default function (eleventyConfig) {
  eleventyConfig.addPlugin(IdAttributePlugin);
  eleventyConfig.addPlugin(syntaxHighlight);
  eleventyConfig.addPassthroughCopy({ "content/css": "css" });

  eleventyConfig.addFilter("year", () => new Date().getFullYear());
  eleventyConfig.addFilter("relativeUrl", (target, current) => {
    if (!target) {
      return "./";
    }

    if (/^(?:[a-z]+:)?\/\//i.test(target) || target.startsWith("mailto:") || target.startsWith("#")) {
      return target;
    }

    const { path: targetPath, suffix } = stripSuffix(target);
    const absoluteTarget = normalizeSitePath(targetPath);
    const absoluteCurrent = normalizeSitePath(current);
    const currentDir = absoluteCurrent.endsWith("/")
      ? absoluteCurrent
      : `${path.posix.dirname(absoluteCurrent)}/`;

    let relative = path.posix.relative(currentDir, absoluteTarget);
    if (!relative) {
      relative = ".";
    }

    if (absoluteTarget.endsWith("/") && !relative.endsWith("/")) {
      relative = `${relative}/`;
    }

    return `${relative}${suffix}`;
  });

  eleventyConfig.addFilter("tutorialPrevious", (tutorials, url) => {
    if (normalizeSitePath(url) === "/tutorials/") {
      return null;
    }

    const index = tutorials.findIndex((tutorial) => normalizeSitePath(tutorial.url) === normalizeSitePath(url));
    if (index > 0) {
      return tutorials[index - 1];
    }

    return index === 0 ? { title: "Tutorials", url: "tutorials/" } : null;
  });
  eleventyConfig.addFilter("tutorialNext", (tutorials, url) => {
    if (normalizeSitePath(url) === "/tutorials/") {
      return tutorials[0] ?? null;
    }

    const index = tutorials.findIndex((tutorial) => normalizeSitePath(tutorial.url) === normalizeSitePath(url));
    return index >= 0 && index < tutorials.length - 1 ? tutorials[index + 1] : null;
  });

  eleventyConfig.addFilter("languageReferencePrevious", (languageReference, url) => {
    if (normalizeSitePath(url) === "/language-reference/") {
      return null;
    }

    const index = languageReference.findIndex((page) => normalizeSitePath(page.url) === normalizeSitePath(url));
    if (index > 0) {
      return languageReference[index - 1];
    }

    return index === 0 ? { title: "Language Reference", url: "language-reference/" } : null;
  });
  eleventyConfig.addFilter("languageReferenceNext", (languageReference, url) => {
    if (normalizeSitePath(url) === "/language-reference/") {
      return languageReference[0] ?? null;
    }

    const index = languageReference.findIndex((page) => normalizeSitePath(page.url) === normalizeSitePath(url));
    return index >= 0 && index < languageReference.length - 1 ? languageReference[index + 1] : null;
  });
}

export const config = {
  markdownTemplateEngine: "njk",
  htmlTemplateEngine: "njk",
  templateFormats: ["md", "njk", "html"],
  dir: {
    input: "content",
    includes: "_includes",
    data: "_data",
    output: "_site"
  }
};
