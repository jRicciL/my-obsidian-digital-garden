# frozen_string_literal: true

Jekyll::Hooks.register [:notes], :pre_render do |note|
  format_tag(note)
end

def format_tag(note)
  note.content.gsub!(
    /\{\{/, 
    "{ {"
  )

  note.content.gsub!(
    /(\B#\w*[a-zA-Z\-]+\w*)/, 
    "<span class='tag'>\\1</span>"
  )
end
